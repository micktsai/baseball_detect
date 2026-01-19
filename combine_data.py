# merge_roboflow_yolov8.py
# -------------------------------------------------
# 需求：
# pip install roboflow pyyaml
# -------------------------------------------------

from dotenv import load_dotenv
from roboflow import Roboflow
import os
import shutil
import yaml

load_dotenv()
# ========== 1. Roboflow 初始化 ==========
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))  # ← 換成你的 API KEY

# Dataset 1
project1 = rf.workspace("mickshelbytsai").project("pitch-tracking-sgse6")
ds1_path = project1.version(3).download("yolov8").location

# Dataset 2
project2 = rf.workspace("mickshelbytsai").project("baseball-detection-2-y6avf")
ds2_path = project2.version(2).download("yolov8").location


# ========== 2. 讀取 class names（確認一致） ==========
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

data1 = load_yaml(f"{ds1_path}/data.yaml")
data2 = load_yaml(f"{ds2_path}/data.yaml")

assert data1["names"] == data2["names"], "❌ Dataset names 不一致，請先確認"


# ========== 3. 建立合併資料夾 ==========
MERGED_ROOT = "merged_dataset"

for split in ["train", "valid", "test"]:
    os.makedirs(f"{MERGED_ROOT}/{split}/images", exist_ok=True)
    os.makedirs(f"{MERGED_ROOT}/{split}/labels", exist_ok=True)


# ========== 4. 合併資料（不改 label，只加 prefix） ==========
def merge_split(src_root, split, prefix):
    img_src = f"{src_root}/{split}/images"
    lbl_src = f"{src_root}/{split}/labels"

    if not os.path.exists(img_src):
        return

    img_dst = f"{MERGED_ROOT}/{split}/images"
    lbl_dst = f"{MERGED_ROOT}/{split}/labels"

    for img in os.listdir(img_src):
        shutil.copy(
            f"{img_src}/{img}",
            f"{img_dst}/{prefix}_{img}"
        )

    for lbl in os.listdir(lbl_src):
        shutil.copy(
            f"{lbl_src}/{lbl}",
            f"{lbl_dst}/{prefix}_{lbl}"
        )


for s in ["train", "valid", "test"]:
    merge_split(ds1_path, s, "ds1")
    merge_split(ds2_path, s, "ds2")


# ========== 5. 產生新的 data.yaml ==========
merged_yaml = {
    "path": MERGED_ROOT,
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": data1["names"]
}

with open(f"{MERGED_ROOT}/data.yaml", "w") as f:
    yaml.dump(merged_yaml, f, sort_keys=False)


# ========== 6. 基本檢查 ==========
def count_files(split):
    img = len(os.listdir(f"{MERGED_ROOT}/{split}/images"))
    lbl = len(os.listdir(f"{MERGED_ROOT}/{split}/labels"))
    print(f"{split}: images={img}, labels={lbl}")

print("\n✅ Merge 完成，檔案數量：")
for s in ["train", "valid", "test"]:
    count_files(s)

print("\n👉 接下來可直接執行：")
print("yolo detect train model=yolov8n.pt data=merged_dataset/data.yaml epochs=100 imgsz=640")
