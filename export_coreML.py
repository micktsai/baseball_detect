import coremltools as ct
import onnx

# 1. 設定你的檔案路徑
onnx_path = "export/baseball_yolov11n_p2_gblur_prune.onnx"  # 你的 ONNX 檔案
coreml_path = "export/baseball_yolov11n_p2_gblur_prune.mlpackage" # 輸出路徑 (新版 CoreML 推薦用 .mlpackage 資料夾格式)

# 2. 載入 ONNX 模型
print(f"正在載入 {onnx_path} ...")
onnx_model = onnx.load(onnx_path)

# 3. 轉換設定
# ⚠️ 關鍵：定義輸入類型為圖片 (ImageType)
# shape: (1, 3, 1280, 1280) -> 根據你之前的設定 1280
# scale: 1/255.0 -> 因為 YOLO 訓練時圖片是 0-255，模型預期 0-1
input_image_type = ct.ImageType(
    name="images",  # 請確認 Netron 中你的輸入節點名稱，YOLO 通常是 "images"
    shape=(1, 3, 1280, 1280), 
    scale=1/255.0, 
    bias=[0, 0, 0]
)

# 4. 執行轉換
print("正在轉換為 CoreML ...")
mlmodel = ct.convert(
    onnx_model,
    inputs=[input_image_type],
    # minimum_deployment_target=ct.target.iOS16, # 可選：指定 iOS 版本
    compute_precision=ct.precision.FLOAT16 # 建議開啟，減少模型大小
)

# 5. 存檔
mlmodel.save(coreml_path)
print(f"🎉 轉換成功！已儲存至 {coreml_path}")