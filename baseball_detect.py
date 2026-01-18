import os
import shutil
import glob
import random
import cv2
import albumentations as A
from roboflow import Roboflow
from ultralytics import YOLO
import torch
from dotenv import load_dotenv

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
# 影像增強（安全版，給小球用）
# ===============================
def augment_dataset(dataset_location, augment_ratio=0.35):
    """
    對 train/images 做「輕量 motion blur」增強
    - 只處理一部分圖片（預設 35%）
    - 不破壞小球形狀
    - label 直接複製
    """
    print(f"🔄 開始本地 augmentation（ratio={augment_ratio}）")

    img_dir = os.path.join(dataset_location, "train", "images")
    lbl_dir = os.path.join(dataset_location, "train", "labels")

    # transform = A.Compose([
    #     # 輕量 motion blur：模擬快門不足（安全）
    #     A.MotionBlur(blur_limit=(7, 15), p=1),

    #     # 非必須，但可幫助亮度差異
    #     A.RandomBrightnessContrast(
    #         brightness_limit=0.1,
    #         contrast_limit=0.1,
    #         p=0.3
    #     ),
    # ])
    transform = A.Compose([
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 5), p=1),
            A.MotionBlur(blur_limit=(5, 7), p=1),
        ], p=0.7),

        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.2
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.15,
            p=0.3
        ),
    ])
    image_paths = (
        glob.glob(os.path.join(img_dir, "*.jpg")) +
        glob.glob(os.path.join(img_dir, "*.png")) +
        glob.glob(os.path.join(img_dir, "*.jpeg"))
    )

    count = 0
    for img_path in image_paths:
        # 只對部分圖片做 augmentation
        if random.random() > augment_ratio:
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        augmented = transform(image=image)["image"]

        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        new_img_name = f"{name}_mb{ext}"
        new_img_path = os.path.join(img_dir, new_img_name)

        cv2.imwrite(new_img_path, augmented)

        # label 直接複製（motion blur 不改幾何）
        old_label = os.path.join(lbl_dir, f"{name}.txt")
        new_label = os.path.join(lbl_dir, f"{name}_mb.txt")

        if os.path.exists(old_label):
            shutil.copy(old_label, new_label)
            count += 1

    print(f"✅ Augmentation 完成，新增 {count} 張訓練圖片")


# ===============================
# 主流程
# ===============================
def main():
    # 避免 CUDA 記憶體殘留
    torch.cuda.empty_cache()
    name = "yolov8n_p2new_mblur_25_pt"
    version = 11
    # ===============================
    # 1. Roboflow 下載資料
    # ===============================
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY")) 
    project = rf.workspace("mickshelbytsai").project("pitch-tracking-sgse6")
    version = project.version(2)
    dataset = version.download("yolov8")
                
    # ===============================
    # 2. 本地 augmentation（安全版）
    # ===============================
    # remove_excess_negatives(dataset.location, remove_ratio=0.9)
    augment_dataset(dataset.location, augment_ratio=0.25)

    # ===============================
    # 3. 建立模型
    # ===============================
    model = YOLO("./yolov8n-p2-new.yaml")
    # ===============================
    # 4. Train（穩定版設定）
    # ===============================
    model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        epochs=200,          # 小球任務，epochs 比 batch 有價值
        patience=30,         # 關鍵：如果 15 輪沒進步就自動停，不用白跑
        imgsz=1280,          # 關鍵：保住小球
        batch=4,             # 8GB/12GB GPU 安全值
        device=0,            # CUDA
        workers=0,           # Windows 穩定必備
        amp=True,           # 避免 cuDNN 不穩
        name=name,
        exist_ok=True,
    )
    '''
    # ===============================
    # 5. Export TFLite（float16）
    # ===============================
    print("🚀 Exporting onnx...")
    model.export(
        format="onnx",
        imgsz=1280,
        nms=False,
        simplify=True
    )
    model.export(
        format="tflite",
        imgsz=1280,
        nms=False,
    )
    # ===============================
    # 6. Copy ONNX output
    # ===============================
    base_path = f"runs/detect/{name}/weights"
    src = f"{base_path}/best.onnx"
    src2 = f"{base_path}/best_saved_model/best_float32.tflite"
    os.makedirs("export", exist_ok=True)
    dst1 = f"export/v{version}/{name}.onnx"
    dst2 = f"export/v{version}/{name}.tflite"

    if os.path.exists(src):
        shutil.copy(src, dst1)
        print(f"🎉 匯出成功：{dst1}")
    else:
        print("❌ 找不到 ONNX 輸出檔，請檢查 weights 資料夾")
    if os.path.exists(src2):
        shutil.copy(src2, dst2)
        print(f"🎉 匯出成功：{dst2}")
    else:
        print("❌ 找不到 TFLite 輸出檔，請檢查 weights 資料夾")
    '''
if __name__ == "__main__":
    main()
