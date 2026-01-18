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
# [核心] 裁切資料集製作函數
# ===============================
def create_crop_dataset(source_path, output_path, crop_size=640, jitter_ratio=0.2, negative_ratio=0.1):
    """
    讀取原始高解析度資料集，生成以球為中心的局部裁切資料集
    """
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    splits = ['train', 'valid', 'test']
    
    # 建立目錄結構
    for split in splits:
        os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)

    print(f"✂️  開始製作裁切資料集...")
    print(f"    來源: {source_path}")
    print(f"    目標: {output_path}")
    print(f"    規格: {crop_size}x{crop_size} (Jitter: {jitter_ratio}, Neg: {negative_ratio})")

    total_crops = 0

    for split in splits:
        img_dir = os.path.join(source_path, split, 'images')
        lbl_dir = os.path.join(source_path, split, 'labels')
        
        if not os.path.exists(img_dir): continue

        img_paths = glob.glob(os.path.join(img_dir, "*"))
        print(f"    正在處理 {split} ({len(img_paths)} 張原圖)...")

        for img_path in tqdm(img_paths):
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
                    s_name = f"{name}_c{i}"
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
                    bg_name = f"{name}_bg"
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
# 影像增強 (針對裁切後的圖片)
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
    
    # 設定名稱 (記得加上 crop 標記以示區別)
    name = "yolov8n_p2_crop640_v1"
    
    # ===============================
    # 1. Roboflow 下載 (Raw Data)
    # ===============================
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY")) 
    project = rf.workspace("mickshelbytsai").project("pitch-tracking-sgse6")
    
    # ⚠️ 請務必在此修改為你的 "No Resize" Version 版本號
    version = project.version(3) 
    print("⬇️ 下載原始資料集...")
    dataset = version.download("yolov8")
    
    # ===============================
    # 2. 執行裁切 (Raw -> Crop)
    # ===============================
    crop_dataset_dir = "dataset_cropped_640"
    create_crop_dataset(
        source_path=dataset.location, 
        output_path=crop_dataset_dir, 
        crop_size=640,      # 這裡設定 640 以符合訓練
        jitter_ratio=0.2,   # 允許中心點偏移 20%
        negative_ratio=0.1  # 10% 背景圖
    )

    # ===============================
    # 3. 執行增強 (On Cropped Data)
    # ===============================
    augment_dataset(crop_dataset_dir, augment_ratio=0.3)

    # ===============================
    # 4. 建立與訓練模型
    # ===============================
    # 既然球變大了，p2 架構依然可以用，但效果會更顯著
    model = YOLO("./yolov8n-p2-new.yaml") 
    
    model.train(
        data=os.path.join(crop_dataset_dir, "data.yaml"), # 指向裁切後的資料
        epochs=150,          
        patience=30,         # 給他多一點耐心
        imgsz=640,           # ⭐️ 關鍵：降回 640，因為這是裁切圖
        batch=16,            # ⭐️ 關鍵：圖變小了，Batch 開大！(試試 16 或 32)
        workers=0,
        amp=True,
        name=name,
        exist_ok=True,
        mosaic=0.0,          
    )

if __name__ == "__main__":
    main()