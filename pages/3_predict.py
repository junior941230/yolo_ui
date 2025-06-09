import streamlit as st
import os, shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# 設定頁面
st.set_page_config(page_title="yolo 預測工具", page_icon="🧮", layout="centered")
st.title("🧮 yolo 預測工具")
st.markdown("使用 YOLO 模型預測圖片")

# 路徑設定
model_root = "runs/detect"
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)

# 取得模型清單
model_dirs = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]
if not model_dirs:
    st.error("❌ 找不到模型資料夾，請確認 `runs/detect` 路徑下有模型。")
    st.stop()

# UI：選擇模型與上傳圖片
col1, col2 = st.columns(2)
with col1:
    option_model = st.selectbox("📦 選擇要使用的模型", model_dirs)
with col2:
    uploaded_files = st.file_uploader(
        "📷 選擇要預測的圖片", accept_multiple_files=True, type=["png", "jpg"]
    )

# 執行預測
if uploaded_files and st.button("🚀 開始預測"):
    # 清空 temp 資料夾
    shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # 載入模型
    model_path = os.path.join(model_root, option_model, "weights", "best.pt")
    if not os.path.exists(model_path):
        st.error(f"❌ 找不到模型權重：{model_path}")
        st.stop()
    model = YOLO(model_path)

    # 儲存上傳圖片
    saved_files = []
    for f in uploaded_files:
        img_path = os.path.join(temp_dir, f.name)
        with open(img_path, "wb") as out_file:
            out_file.write(f.read())
        saved_files.append(img_path)

    # 預測並顯示結果
    st.subheader("📊 預測結果")
    for i, img_path in enumerate(saved_files):
        results = model(img_path)
        result_img_path = os.path.join(temp_dir, f"result_{i}.jpg")
        results[0].save(filename=result_img_path)
        st.image(result_img_path, caption=f"預測圖：{Path(img_path).name}", use_container_width =True)
        with open(result_img_path, "rb") as f:
            st.download_button(
                label="📥 下載預測圖",
                data=f,
                file_name=Path(result_img_path).name,
                mime="image/jpeg"
            )