import streamlit as st
import os, shutil
import subprocess
from pathlib import Path
from ultralytics import YOLO

# 設定頁面
st.set_page_config(page_title="NCNN int8 量化工具", page_icon="🧮", layout="centered")
st.title("🧮 NCNN int8 量化工具")
st.markdown("將 YOLO 模型轉為 NCNN 並進行 int8 校正量化")

# 路徑設定
model_root = "runs/detect"
dataset_root = "datasets"
temp_dir = "temp"

# 取得模型與資料集清單
model_dirs = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]
dataset_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]

# UI：選擇模型與資料集
col1, col2 = st.columns(2)
with col1:
    option_model = st.selectbox("📦 選擇要量化的模型", model_dirs)
with col2:
    option_data_set = st.selectbox("🖼️ 選擇校正圖片資料集", dataset_dirs)

# 開始量化
if st.button("🔧 開始量化"):
    st.info("🔄 開始模型匯出與量化...請稍候")

    # 清除 temp 目錄
    shutil.rmtree(f"{temp_dir}/ncnn_model", ignore_errors=True)
    shutil.rmtree(f"{temp_dir}/ncnn_quantize", ignore_errors=True)
    os.makedirs(f"{temp_dir}/ncnn_model", exist_ok=True)
    os.makedirs(f"{temp_dir}/ncnn_quantize", exist_ok=True)

    # 匯出 YOLO 模型為 NCNN
    model_path = os.path.join(model_root, option_model, "weights", "best.pt")
    model = YOLO(model_path)
    model.export(format="ncnn")

    # 複製匯出檔案到 temp
    export_dir = os.path.join(model_root, option_model, "weights", "best_ncnn_model")
    copydatas = os.listdir(export_dir)
    for data_name in copydatas:
        shutil.copy(os.path.join(export_dir, data_name), f"{temp_dir}/ncnn_model/{data_name}")

    # 產生 calibration.txt
    calib_txt_path = f"{temp_dir}/calibration.txt"
    train_image_dir = os.path.join(dataset_root, option_data_set, "images", "train")

    if not os.path.exists(train_image_dir):
        st.error(f"❗ 找不到訓練圖片資料夾：{train_image_dir}")
        st.stop()

    with open(calib_txt_path, "w") as f:
        for file in sorted(os.listdir(train_image_dir)):
            if file.lower().endswith((".jpg", ".png")):
                img_path = os.path.abspath(os.path.join(train_image_dir, file))
                f.write(img_path + "\n")

    # 執行 ncnn2int8 命令
    cmd = [
        "./ncnn2int8",
        f"{temp_dir}/ncnn_model/model.param",
        f"{temp_dir}/ncnn_model/model.bin",
        f"{temp_dir}/ncnn_quantize/quantize_model.param",
        f"{temp_dir}/ncnn_quantize/quantize_model.bin",
        calib_txt_path
    ]

    with st.spinner("📦 執行 ncnn2int8 中..."):
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 顯示命令與輸出結果
    with st.expander("📜 執行命令與日誌"):
        st.code(" ".join(cmd), language="bash")
        st.text_area("stdout", result.stdout, height=200)
        st.text_area("stderr", result.stderr, height=100)

    # 成功提示與下載按鈕
    quant_param_path = f"{temp_dir}/ncnn_quantize/quantize_model.param"
    quant_bin_path = f"{temp_dir}/ncnn_quantize/quantize_model.bin"

    if os.path.exists(quant_param_path) and os.path.exists(quant_bin_path):
        st.success("✅ 量化完成！以下為下載連結：")
        with open(quant_param_path, "rb") as f:
            st.download_button("📥 下載 quantize_model.param", f, file_name="quantize_model.param")
        with open(quant_bin_path, "rb") as f:
            st.download_button("📥 下載 quantize_model.bin", f, file_name="quantize_model.bin")
    else:
        st.error("❌ 量化失敗，請檢查模型與圖片是否正確")
