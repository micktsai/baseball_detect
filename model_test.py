import cv2
import numpy as np
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Error: neither 'tflite_runtime' nor 'tensorflow' is installed.")
        print("Please install one of them: pip install tflite-runtime OR pip install tensorflow")
        exit(1)
import os
import argparse
import glob
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# 參數設定
# ==========================================
MODEL_PATH = "models/yolo8n_p2new/yolov8n_p2new_mblur_40_db_saved_model/yolov8n_p2new_mblur_40_db_float16.tflite"
OUTPUT_BASE_DIR = "results"

ROI_WIDTH = 480
ROI_HEIGHT = 320

# 顏色定義 (BGR)
ROI_COLOR = (0, 255, 0)    # Green
BOX_COLOR = (0, 0, 255)    # Red

# 狀態定義
STATE_SEARCH = 0
STATE_TRACK = 1

def get_video_path():
    """解析 CLI 參數並回傳影片路徑"""
    parser = argparse.ArgumentParser(description="Baseball Ball Detection & Tracking")
    parser.add_argument("-v", "--video", type=str, help="Video filename (in videos/ folder), e.g., 'monster1'")
    args = parser.parse_args()
    
    default_video = "videos/monster2.MP4"
    
    if args.video:
        # 1. 檢查是否直接是路徑
        if os.path.exists(args.video):
            return args.video
            
        # 2. 檢查 videos/ 下的完整檔名
        path_in_videos = os.path.join("videos", args.video)
        if os.path.exists(path_in_videos):
            return path_in_videos
            
        # 3. 嘗試搜尋 videos/ 下的同名檔案 (忽略副檔名)
        candidates = glob.glob(os.path.join("videos", f"{args.video}.*"))
        if candidates:
            # 優先找 mp4, MP4, mov, MOV
            print(f"🔍 Found candidates: {candidates}")
            return candidates[0]
            
        print(f"❌ Error: Video '{args.video}' not found in videos/ directory.")
        exit(1)
    else:
        return default_video

def load_models():
    """載入 YOLO-Pose 和 TFLite 模型"""
    # 初始化 Pose 模型 (YOLOv8-Pose)
    print("🚀 正在載入 Pose 模型: yolov8n-pose.pt")
    pose_model = YOLO('yolov8n-pose.pt')
    
    # 載入 TFLite 模型
    print(f"🚀 正在載入 Ball Detection 模型: {MODEL_PATH}")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return pose_model, interpreter

def get_roi(center_x, center_y, frame_width, frame_height):
    """計算 ROI 座標，包含了邊界檢查"""
    x1 = max(0, center_x - ROI_WIDTH // 2)
    y1 = max(0, center_y - ROI_HEIGHT // 2)
    x2 = min(frame_width, center_x + ROI_WIDTH // 2)
    y2 = min(frame_height, center_y + ROI_HEIGHT // 2)
    
    # 確保 ROI 大小固定 (除了邊界)
    if x2 - x1 < ROI_WIDTH:
        if x1 == 0:
            x2 = min(frame_width, x1 + ROI_WIDTH)
        else:
            x1 = max(0, x2 - ROI_WIDTH)
            
    if y2 - y1 < ROI_HEIGHT:
        if y1 == 0:
            y2 = min(frame_height, y1 + ROI_HEIGHT)
        else:
            y1 = max(0, y2 - ROI_HEIGHT)
            
    return int(x1), int(y1), int(x2), int(y2)

def analyze_pose(frame, pose_model):
    """執行骨架分析，回傳右手位置 (x, y) 或 (-1, -1)"""
    pose_results = pose_model(frame, verbose=False)
    
    hand_x, hand_y = -1, -1
    
    if pose_results and len(pose_results[0].keypoints) > 0:
        keypoints = pose_results[0].keypoints
        if keypoints is not None and keypoints.conf is not None:
             # COCO Keypoint Index 10 is Right Wrist
            rw_idx = 10
            
            # 取得 Right Wrist 資料
            if keypoints.xy.shape[1] > rw_idx:
                rw_x = keypoints.xy[0][rw_idx][0].item()
                rw_y = keypoints.xy[0][rw_idx][1].item()
                rw_conf = keypoints.conf[0][rw_idx].item()
                
                if rw_conf > 0.5:
                    hand_x, hand_y = int(rw_x), int(rw_y)
    
    return hand_x, hand_y

def run_tflite_inference(frame_roi, interpreter):
    """在 ROI 上執行 TFLite 模型推論"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'] # [1, 640, 640, 3]

    input_h, input_w = input_shape[1], input_shape[2]
    
    # 預處理
    img = cv2.resize(frame_roi, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # 推論
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    # 解析輸出 (假設輸出格式 [1, 5, 8400])
    output_data = interpreter.get_tensor(output_details[0]['index'])[0] 
    output_data = output_data.T
    
    boxes = []
    scores = []
    
    # 閾值過濾
    conf_threshold = 0.45 
    
    for i in range(len(output_data)):
        row = output_data[i]
        score = row[4] # class score (ball)
        
        if score > conf_threshold:
            # YOLO output: cx, cy, w, h (normalized relative to 640x640)
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            
            # 轉回 ROI 座標 (480x320)
            # 因為我們 resize 成 640x640 丟進去，所以要還原比例
            scale_x = frame_roi.shape[1] / input_w
            scale_y = frame_roi.shape[0] / input_h
            
            x1 = int((cx - w/2) * input_w * scale_x)
            y1 = int((cy - h/2) * input_h * scale_y)
            x2 = int((cx + w/2) * input_w * scale_x)
            y2 = int((cy + h/2) * input_h * scale_y)
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            
    return boxes, scores

def process_video(video_path):
    """主影片處理流程"""
    pose_model, interpreter = load_models()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 準備輸出目錄
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = "yolov8n_p2new_mblur_40_db_float16" # 簡化名稱
    result_dir = os.path.join(OUTPUT_BASE_DIR, model_name, video_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # 清空舊結果
    for f in os.listdir(result_dir):
        os.remove(os.path.join(result_dir, f))
        
    print(f"🎞️ 開始處理影片，共 {total_frames} 幀 ({frame_width}x{frame_height})...")
    
    frame_idx = 1
    pbar = tqdm(total=total_frames, desc="Processing Frames")
    
    # 狀態變數
    current_state = STATE_SEARCH
    
    last_ball_x = 0
    last_ball_y = 0
    roi_direction = 0 # 1: right, -1: left
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, 0, 0
        current_roi_frame = None
        ball_detected = False
        
        annotated_frame = frame.copy()

        # ---------------------------------------------------------
        # 1. 狀態機：SEARCH 模式
        # ---------------------------------------------------------
        if current_state == STATE_SEARCH:
            hand_center_x, hand_center_y = analyze_pose(frame, pose_model)
            
            if hand_center_x != -1:
                # 視覺化右手
                cv2.circle(annotated_frame, (hand_center_x, hand_center_y), 8, (0, 255, 255), -1)
                cv2.putText(annotated_frame, "Right Hand", (hand_center_x + 10, hand_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # 決定初始方向：手在球的左邊(畫面左側) -> ROI 往右走(遠離手)
                if hand_center_x < frame_width / 2:
                    roi_direction = 1 # 往右 (遠離手)
                else:
                    roi_direction = -1 # 往左 (遠離手)
                
                # 抓取右手附近的 ROI
                roi_x1, roi_y1, roi_x2, roi_y2 = get_roi(hand_center_x, hand_center_y, frame_width, frame_height)
                current_roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                
                # 畫出搜尋 ROI
                cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), ROI_COLOR, 2)
                cv2.putText(annotated_frame, "Searching Ball (Right Hand)", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 2)

        # ---------------------------------------------------------
        # 2. 狀態機：TRACK 模式
        # ---------------------------------------------------------
        elif current_state == STATE_TRACK:
            # 追蹤模式下，ROI 根據球的最後和方向外推
            # 策略：以球為中心，往「遠離手」的方向推 50 px (Leading ROI)
            roi_center_x = last_ball_x + (50 * roi_direction)
            roi_center_y = last_ball_y 
            
            roi_x1, roi_y1, roi_x2, roi_y2 = get_roi(roi_center_x, roi_center_y, frame_width, frame_height)
            current_roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # 畫出追蹤 ROI
            cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), ROI_COLOR, 2)
            cv2.putText(annotated_frame, "Tracking Ball (Sticky)", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 2)

        # ---------------------------------------------------------
        # 3. 執行球體偵測 (TFLite)
        # ---------------------------------------------------------
        if current_roi_frame is not None and current_roi_frame.size > 0:
            boxes, scores = run_tflite_inference(current_roi_frame, interpreter)
            
            best_score = 0
            best_box = None
            
            if len(boxes) > 0:
                # 找出最高分的
                idx = np.argmax(scores)
                best_score = scores[idx]
                box = boxes[idx]
                
                # 轉換回全域座標
                x1, y1, x2, y2 = box
                abs_x1 = x1 + roi_x1
                abs_y1 = y1 + roi_y1
                abs_x2 = x2 + roi_x1
                abs_y2 = y2 + roi_y1
                
                best_box = [abs_x1, abs_y1, abs_x2, abs_y2]
                
                # 畫出球
                label = f"ball {best_score:.2f}"
                cv2.rectangle(annotated_frame, (abs_x1, abs_y1), (abs_x2, abs_y2), BOX_COLOR, 2)
                cv2.putText(annotated_frame, label, (abs_x1, abs_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # ---------------------------------------------------------
            # 4. 更新狀態
            # ---------------------------------------------------------
            if best_box:
                # 偵測到球
                ball_detected = True
                bx1, by1, bx2, by2 = best_box
                current_ball_x = (bx1 + bx2) / 2
                current_ball_y = (by1 + by2) / 2
                
                last_ball_x = current_ball_x
                last_ball_y = current_ball_y
                
                # 如果是從 Search 轉 Track，決定追蹤方向並檢查手中球
                if current_state == STATE_SEARCH:
                    # 計算球與手的距離 (需要 hand_center_x, hand_center_y from prev step)
                    # 注意：在 SEARCH 模式下 hand_center_x 是有定義的
                    dist_to_hand = ((current_ball_x - hand_center_x)**2 + (current_ball_y - hand_center_y)**2)**0.5
                    
                    if dist_to_hand < 15: # User suggested 15px logic check from edit, previously was 5px
                         # 球離手太近，視為還在手中
                        print(f"Frame {frame_idx}: Ball close to hand ({dist_to_hand:.1f}px). Staying in SEARCH.")
                        # 保持不變，繼續搜尋
                    else:
                        # 距離夠遠，視為投出
                        if current_ball_x > hand_center_x:
                            roi_direction = 1 # 往右
                        else:
                            roi_direction = -1 # 往左
                        
                        print(f"Frame {frame_idx}: Released! Dist: {dist_to_hand:.1f}, Dir: {roi_direction}")
                        current_state = STATE_TRACK
                
                else:
                    # 已經在 Track 模式，就繼續 Track
                    current_state = STATE_TRACK
            
            else:
                # 該 ROI 沒找到球 (可能是跟丟或被遮擋)
                if current_state == STATE_TRACK:
                    # 固定外推 (Fixed Extrapolation)
                    last_ball_x += (100 * roi_direction)
                    # 保持 Track
                    current_state = STATE_TRACK
                else:
                    # Search 模式沒找到 -> 繼續 Search
                    current_state = STATE_SEARCH

        # 儲存圖片
        output_filename = f"img_result{frame_idx}.jpg"
        output_path = os.path.join(result_dir, output_filename)
        cv2.imwrite(output_path, annotated_frame)

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f"\n✅ 測試完成！所有結果已存至 {result_dir}")

if __name__ == "__main__":
    video_path = get_video_path()
    process_video(video_path)
