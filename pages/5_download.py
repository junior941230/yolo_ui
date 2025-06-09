import streamlit as st
import os

# 設定頁面
st.set_page_config(page_title="倉庫下載區", page_icon="🧮", layout="centered")
st.title("🧮 Yolo倉庫下載")

# 路徑設定
pt_model_root = "runs/detect"

# 取得模型與資料集清單
model_dirs = [d for d in os.listdir(pt_model_root) if os.path.isdir(os.path.join(pt_model_root, d))]

# UI：選擇模型與資料集
col1, col2 = st.columns(2)
with col1:
    option_model = st.selectbox("📦 選擇要下載的模型", model_dirs)
    model_dir =  os.path.join(pt_model_root,option_model,"weights/best.pt")
    print(model_dir)
    with open(model_dir, "rb") as f:
        st.download_button("📥 下載 best.bt", f, file_name="best.bt")
