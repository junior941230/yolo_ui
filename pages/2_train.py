import streamlit as st
import os
import time
from ultralytics import YOLO
from PIL import Image

# 頁面設定
st.set_page_config(page_title="YOLO 訓練平台", page_icon="📈", layout="centered")

# 資料夾設定
dataset_root = "datasets"
run_output_root = "runs/detect"

st.title("🧠 YOLO 訓練介面")
st.markdown("請選擇已整理好的資料集，設定參數後即可開始訓練 YOLO 模型。")

# 檢查資料集是否存在
data_set_paths = [
    d for d in os.listdir(dataset_root)
    if os.path.isdir(os.path.join(dataset_root, d)) and
       os.path.exists(os.path.join(dataset_root, d, "data.yaml"))
]

if not data_set_paths:
    st.error("❗ 找不到任何可用的資料集，請先建立 datasets/{name}/data.yaml。")
    st.stop()

# === 設定區塊 ===
with st.expander("⚙️ 訓練參數設定", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        option = st.selectbox("📁 選擇要訓練的資料集", data_set_paths)
    with col2:
        epoch = st.slider("📌 訓練 Epoch 數量", 1, 200, 100)

    result_name = st.text_input("📂 訓練結果資料夾名稱", value=option)

    st.info(f"📄 訓練使用的設定檔：`{dataset_root}/{option}/data.yaml`")

# === 訓練函式 ===
def train_model(dataset_path, epoch, output_name):
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=os.path.join(dataset_root, dataset_path, "data.yaml"),
        epochs=epoch,
        imgsz=640,
        device=0,
        name=output_name
    )
    return results

# === 按鈕啟動訓練 ===
if st.button("🚀 開始訓練 YOLO 模型"):
    with st.spinner("模型訓練中，請稍候..."):
        start_time = time.time()
        results = train_model(option, epoch, result_name)
        duration = time.time() - start_time

    st.success(f"✅ 模型訓練完成！總耗時：{duration:.2f} 秒")
    st.markdown("---")

    # 顯示混淆矩陣圖
    conf_matrix_path = os.path.join(run_output_root, result_name, "confusion_matrix_normalized.png")
    if os.path.exists(conf_matrix_path):
        st.markdown("### 📊 混淆矩陣")
        st.image(Image.open(conf_matrix_path), caption="Normalized Confusion Matrix", use_column_width=True)
    else:
        st.warning("⚠️ 找不到混淆矩陣圖，可能是 Epoch 太少或訓練中未產生。")

    # 顯示訓練資料夾位置
    st.info(f"📂 訓練結果儲存於：`{os.path.join(run_output_root, result_name)}`")
