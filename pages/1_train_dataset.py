import streamlit as st
from PIL import Image
import os
import shutil
from pathlib import Path
from convert_data_set_albumentations import convert_data_set  # <-- 請確保改為新版檔案名

st.set_page_config(page_title="YOLO 資料集上傳與轉換工具", page_icon="🐍", layout="centered")

# 頁首
st.title("🧰 YOLO 資料集上傳與增強轉換工具")
st.markdown("這是一個工具，可將 `imageXXX.png`、`imageXXX.txt` 與 `classes.txt` 整理並可選擇進行資料增強為 YOLO 訓練資料夾")

# 資料夾路徑
upload_dir = "temp_data"
os.makedirs(upload_dir, exist_ok=True)

# 初始化 session_state
if "clear_upload" not in st.session_state:
    st.session_state.clear_upload = False

# === 區塊 1：上傳與清除 ===
with st.expander("📤 上傳圖片與標註資料", expanded=True):
    st.info("請上傳 `imageXXX.png`, `imageXXX.txt` 及 `classes.txt`")
    if st.button("❌ 清除已上傳的資料"):
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
            os.makedirs(upload_dir, exist_ok=True)
        st.session_state.clear_upload = True
        st.warning("已清除所有上傳的檔案，請重新上傳。")

    if not st.session_state.clear_upload:
        uploaded_files = st.file_uploader(
            "選擇檔案",
            accept_multiple_files=True,
            type=["png", "jpg", "txt"],
            key="uploader",
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"✅ 成功上傳 {len(uploaded_files)} 個檔案！")

    else:
        st.info("目前已取消所有上傳資料，請重新整理頁面以上傳新檔案。")

# === 區塊 2：轉換參數與執行 ===
if len(os.listdir(upload_dir)) != 0:
    st.markdown("---")
    st.subheader("⚙️ 資料集轉換參數")
    val_percent =0
    col1, col2 = st.columns(2)
    with col1:
        output_name = st.text_input("📂 輸出資料集名稱", value="yolo_train_data")
    with col2:
        val_mode = st.radio("📋 驗證集選擇方式", ["隨機分割", "手動勾選"])
    if val_mode == "隨機分割":
        with col2:
            val_percent = st.slider("📊 驗證集比例（隨機）", 0.05, 0.5, 0.2, step=0.05)
            val_selected = None
    else:
        st.markdown("## 📸 點選圖片來加入驗證集")
        image_files = sorted([f for f in os.listdir(upload_dir) if f.endswith((".png", ".jpg"))])
        image_names = sorted([Path(f).stem for f in image_files])

        # 初始化 session_state 儲存選擇狀態
        if "val_selected_dict" not in st.session_state:
            st.session_state.val_selected_dict = {name: False for name in image_names}

        cols = st.columns(5)  # 每行顯示 5 張

        for idx, img_name in enumerate(image_names):
            ext = ".png" if os.path.exists(os.path.join(upload_dir, f"{img_name}.png")) else ".jpg"
            img_path = os.path.join(upload_dir, f"{img_name}{ext}")
            image = Image.open(img_path)

            with cols[idx % 5]:
                st.image(image, caption=img_name, use_container_width=True)
                st.session_state.val_selected_dict[img_name] = st.checkbox(
                    "作為驗證集", key=f"check_{img_name}", value=st.session_state.val_selected_dict[img_name]
                )

        # 將勾選的名稱彙整為 list
        val_selected = [name for name, checked in st.session_state.val_selected_dict.items() if checked]

    col1, col2 = st.columns([1, 2])
    with col1:
        augment = st.checkbox("📈 啟用資料增強", value=True)
    with col2:
        augment_times = st.slider("🔁 每張圖片增強幾次", 1, 20, 4)

    if st.button("🚀 開始轉換資料集"):
        classes_path = os.path.join(upload_dir, "classes.txt")
        if not os.path.exists(classes_path):
            st.error("❗ 找不到 classes.txt，請確認有上傳。")
        elif val_mode == "手動勾選" and (val_selected is None or len(val_selected) == 0):
            st.error("❗ 請選擇至少一張驗證集圖片。")
        else:
            output_path = os.path.join("datasets", output_name)
            if os.path.exists(output_path):
                st.error(f"❗ 資料夾 `{output_name}` 已存在，請改名或稍後再試。")
            else:
                with st.spinner("資料轉換中..."):
                    train_count, val_count = convert_data_set(
                        output_name=output_name,
                        augment=augment,
                        augment_times=augment_times,
                        val_percent=val_percent,
                        val_list=val_selected
                    )
                st.success(f"🎉 資料集已建立於 `{output_path}`")
                st.dataframe({
                    "類型": ["訓練集", "驗證集"],
                    "樣本數": [train_count, val_count]
                })