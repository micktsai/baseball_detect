import os
import shutil
import glob
import random
import cv2
import yaml
import albumentations as A
from roboflow import Roboflow
from ultralytics import YOLO
import torch
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()
# ===============================
# [NEW] 移除過多無球負樣本
# ===============================
def remove_excess_negatives(dataset_location, remove_ratio=0.8):
    """
    移除 train set 中過多的「沒有標註（無球）」圖片
    - remove_ratio=0.8 → 移除 80% 無球圖片
    - 有球圖片 100% 保留
    """
    print(f"🧹 移除過多負樣本（ratio={remove_ratio}）")

    img_dir = os.path.join(dataset_location, "train", "images")
    lbl_dir = os.path.join(dataset_location, "train", "labels")

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(img_dir, ext)))

    empty_samples = []

    for img_path in image_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(lbl_dir, f"{name}.txt")

        # 判斷是否為「無球圖片」
        if not os.path.exists(label_path):
            empty_samples.append((img_path, None))
        elif os.path.getsize(label_path) == 0:
            empty_samples.append((img_path, label_path))

    if not empty_samples:
        print("    - 沒有發現無球圖片，跳過")
        return

    remove_count = int(len(empty_samples) * remove_ratio)
    to_remove = random.sample(empty_samples, remove_count)

    for img_path, label_path in to_remove:
        if os.path.exists(img_path):
            os.remove(img_path)
        if label_path and os.path.exists(label_path):
            os.remove(label_path)

    print(f"    - 發現 {len(empty_samples)} 張無球圖片")
    print(f"    - 已移除 {remove_count} 張")
# ===============================
# [核心] 裁切資料集製作函數 (已修改：支援單一或多個來源)
# ===============================
def create_crop_dataset(source_input, output_path, crop_size=640, jitter_ratio=0.2, negative_ratio=0.1):
    """
    讀取原始高解析度資料集，生成以球為中心的局部裁切資料集
    Args:
        source_input: 
            - 可以是單一路徑字串 "path/to/dataset"
            - 也可以是列表 [("path/to/ds1", "prefix1"), ("path/to/ds2", "prefix2")]
    """
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    # 1. 統一輸入格式
    sources = []
    if isinstance(source_input, str):
        sources = [(source_input, "")] # 單一模式，無前綴
    elif isinstance(source_input, list):
        sources = source_input         # 合併模式
    
    splits = ['train', 'valid', 'test']
    
    # 建立目錄結構
    for split in splits:
        os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)

    print(f"✂️  開始製作裁切資料集...")
    print(f"    來源數量: {len(sources)}")
    print(f"    目標: {output_path}")
    print(f"    規格: {crop_size}x{crop_size} (Jitter: {jitter_ratio}, Neg: {negative_ratio})")

    total_crops = 0

    # 2. 遍歷所有來源資料集
    for source_path, prefix in sources:
        if not os.path.exists(source_path):
            print(f"    ⚠️ 找不到來源: {source_path}，跳過")
            continue
            
        print(f"    📂 處理資料集: {prefix} ({source_path})")

        for split in splits:
            img_dir = os.path.join(source_path, split, 'images')
            lbl_dir = os.path.join(source_path, split, 'labels')
            
            if not os.path.exists(img_dir): continue

            img_paths = glob.glob(os.path.join(img_dir, "*"))
            
            # 使用 tqdm 顯示進度
            for img_path in tqdm(img_paths, desc=f"       {split}", leave=False):
                filename = os.path.basename(img_path)
                name, ext = os.path.splitext(filename)
                lbl_path = os.path.join(lbl_dir, f"{name}.txt")

                image = cv2.imread(img_path)
                if image is None: continue
                h_img, w_img, _ = image.shape

                boxes = [] 
                if os.path.exists(lbl_path):
                    with open(lbl_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])

                # --- 1. 正樣本 (有球) ---
                for i, box in enumerate(boxes):
                    cls, xc, yc, bw, bh = box
                    abs_xc, abs_yc = xc * w_img, yc * h_img
                    
                    # 隨機偏移 (模擬追蹤誤差)
                    offset_limit = crop_size * jitter_ratio
                    off_x = random.uniform(-offset_limit, offset_limit)
                    off_y = random.uniform(-offset_limit, offset_limit)
                    
                    crop_cx = abs_xc + off_x
                    crop_cy = abs_yc + off_y
                    
                    x1 = int(crop_cx - crop_size / 2)
                    y1 = int(crop_cy - crop_size / 2)
                    
                    # 邊界限制
                    x1 = max(0, min(x1, w_img - crop_size))
                    y1 = max(0, min(y1, h_img - crop_size))
                    x2, y2 = x1 + crop_size, y1 + crop_size

                    crop_img = image[y1:y2, x1:x2]
                    
                    # 轉換 Label
                    new_labels = []
                    for b in boxes:
                        b_cls, b_xc, b_yc, b_bw, b_bh = b
                        b_abs_x, b_abs_y = b_xc * w_img, b_yc * h_img
                        b_abs_w, b_abs_h = b_bw * w_img, b_bh * h_img
                        
                        if x1 < b_abs_x < x2 and y1 < b_abs_y < y2:
                            n_xc = (b_abs_x - x1) / crop_size
                            n_yc = (b_abs_y - y1) / crop_size
                            n_bw = b_abs_w / crop_size
                            n_bh = b_abs_h / crop_size
                            
                            # Clip 0-1
                            n_xc = max(0, min(1, n_xc))
                            n_yc = max(0, min(1, n_yc))
                            n_bw = max(0, min(1, n_bw))
                            n_bh = max(0, min(1, n_bh))
                            
                            new_labels.append(f"{b_cls} {n_xc:.6f} {n_yc:.6f} {n_bw:.6f} {n_bh:.6f}")

                    if new_labels:
                        # [修改點] 加上前綴避免檔名衝突
                        p_str = f"{prefix}_" if prefix else ""
                        s_name = f"{p_str}{name}_c{i}"
                        
                        cv2.imwrite(os.path.join(output_path, split, 'images', f"{s_name}.jpg"), crop_img)
                        with open(os.path.join(output_path, split, 'labels', f"{s_name}.txt"), 'w') as f:
                            f.write("\n".join(new_labels))
                        total_crops += 1

                # --- 2. 負樣本 (隨機背景) ---
                if random.random() < negative_ratio:
                    rx = random.randint(0, max(1, w_img - crop_size))
                    ry = random.randint(0, max(1, h_img - crop_size))
                    
                    # 檢查是否有球
                    has_ball = False
                    for b in boxes:
                        bx, by = b[1] * w_img, b[2] * h_img
                        if rx < bx < rx + crop_size and ry < by < ry + crop_size:
                            has_ball = True; break
                    
                    if not has_ball:
                        bg_crop = image[ry:ry+crop_size, rx:rx+crop_size]
                        
                        # [修改點] 加上前綴
                        p_str = f"{prefix}_" if prefix else ""
                        bg_name = f"{p_str}{name}_bg"
                        
                        cv2.imwrite(os.path.join(output_path, split, 'images', f"{bg_name}.jpg"), bg_crop)
                        open(os.path.join(output_path, split, 'labels', f"{bg_name}.txt"), 'w').close()
                        total_crops += 1
    
    print(f"✅ 裁切完成！共生成 {total_crops} 張圖片。")

    # 建立 data.yaml
    yaml_content = {
        'path': os.path.abspath(output_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['baseball']
    }
    with open(os.path.join(output_path, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f)


# ===============================
# 影像增強 (完全不動)
# ===============================
def augment_dataset(dataset_location, augment_ratio=0.35):
    """
    對裁切後的 train/images 做 motion blur
    """
    print(f"🔄 開始 Augmentation (Target: {dataset_location}, Ratio={augment_ratio})")

    img_dir = os.path.join(dataset_location, "train", "images")
    lbl_dir = os.path.join(dataset_location, "train", "labels")

    transform = A.Compose([
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 5), p=1), # 小模糊
            A.MotionBlur(blur_limit=(5, 7), p=1), # 中模糊
        ], p=0.7),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.3),
    ])

    image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    
    count = 0
    for img_path in image_paths:
        if random.random() > augment_ratio: continue

        image = cv2.imread(img_path)
        if image is None: continue

        augmented = transform(image=image)["image"]
        
        name, ext = os.path.splitext(os.path.basename(img_path))
        new_path = os.path.join(img_dir, f"{name}_mb{ext}")
        cv2.imwrite(new_path, augmented)

        # 複製 Label
        old_lbl = os.path.join(lbl_dir, f"{name}.txt")
        new_lbl = os.path.join(lbl_dir, f"{name}_mb.txt")
        if os.path.exists(old_lbl):
            shutil.copy(old_lbl, new_lbl)
            count += 1

    print(f"✅ Augmentation 完成，新增 {count} 張模糊樣本")


# ===============================
# 主流程
# ===============================
def main():
    torch.cuda.empty_cache()
    
    # 設定名稱
    name = "yolov8n_p2_mblur3-7_35_p50_pt_bd2"
    '''
    # ===============================
    # 1. Roboflow 下載 (合併邏輯)
    # ===============================
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY")) 
    
    # 資料集 1
    project1 = rf.workspace("mickshelbytsai").project("pitch-tracking-sgse6")
    ds1_path = project1.version(3).download("yolov8").location
    
    # 資料集 2 (範例：你可以填入你的第二個 project)
    project2 = rf.workspace("mickshelbytsai").project("baseball-detection-2-y6avf")
    ds2_path = project2.version(2).download("yolov8").location
    
    # 建立來源清單 [(路徑, 前綴代號)]
    source_list = [
        (ds1_path, "ds1"),
        (ds2_path, "ds2"), # 如果有第二個就解開這行
    ]
    
    # ===============================
    # 2. 執行裁切 (傳入 List)
    # ===============================
    crop_dataset_dir = "dataset_cropped_640s"
    create_crop_dataset(
        source_input=source_list,   # <--- 這裡改成傳入 list
        output_path=crop_dataset_dir, 
        crop_size=640,
        jitter_ratio=0.2, 
        negative_ratio=0.1 
    )
    '''
    crop_dataset_dir = "merged_dataset"
    remove_excess_negatives(crop_dataset_dir, remove_ratio=0.8)
    # ===============================
    # 3. 執行增強 (不變)
    # ===============================
    augment_dataset(crop_dataset_dir, augment_ratio=0.35)

    # ===============================
    # 4. 建立與訓練模型 (不變)
    # ===============================
    model = YOLO("./yolov8n-p2-new.yaml") 
    
    model.train(
        data=os.path.join(crop_dataset_dir, "data.yaml"),
        epochs=200,          
        patience=50,
        imgsz=1280,
        batch=8,
        workers=0,
        amp=True,
        name=name,
        exist_ok=True,
        mosaic=0.0,
    )

if __name__ == "__main__":
    main()