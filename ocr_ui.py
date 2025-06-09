import streamlit as st

st.set_page_config(
    page_title="YOLO 模型整合 UI",
    page_icon="👋",
    layout="centered",
)

st.title("👋 YOLO 模型整合工具 UI")
st.markdown("""
歡迎使用 **YOLO 模型整合平台**，本工具整合了三大功能模組：

---

## 🧰 功能說明

### 1️⃣ 資料集上傳與格式轉換
📂 功能位置：`資料集轉換工具` 頁面  
🔧 目的：將自訂圖像（imageXXX.png）、標註（imageXXX.txt）和類別（classes.txt）轉換為 YOLO 格式的訓練資料夾。  
📦 輸出結果：會建立 `datasets/你的資料夾名稱/`，包含訓練與驗證圖片與標註。

👉 支援功能：
- 自訂訓練/驗證比例
- 勾選是否旋轉圖像增強（augmentation）
- 會顯示轉換結果筆數（以表格呈現）

---

### 2️⃣ YOLO 模型訓練
🚀 功能位置：`模型訓練` 頁面  
🏋️‍♂️ 目的：選擇一個轉換好的資料集進行 YOLOv8 訓練。  
📂 訓練結果將儲存在：`runs/detect/訓練名稱/`

👉 可設定：
- 訓練 epoch 數量（滑桿調整）
- 輸出資料夾名稱
- 顯示訓練耗時與混淆矩陣（如有產生）

---

### 3️⃣ NCNN int8 量化工具
🧮 功能位置：`NCNN量化` 頁面  
🎯 目的：將訓練完成的 YOLO 模型轉為 NCNN 並使用自定圖片進行 int8 量化。

👉 步驟說明：
- 選擇訓練過的 YOLO 模型
- 選擇資料集作為校正用圖片（自動讀取 `images/train` 資料夾）
- 自動產生 calibration.txt 並執行 `ncnn2int8`
- 產生並提供 `quantize_model.param` 與 `quantize_model.bin` 檔案下載

---

## 📎 注意事項

- 所有圖片命名需為 `image001.png` 搭配 `image001.txt`
- 請確認 `classes.txt` 檔案有上傳
- 若有新增模型或資料集，請重新整理頁面以更新選項
- 若無法成功執行 `ncnn2int8`，請確認權限並編譯成功

---

## 🙌 建議使用流程

1. 🔼 **進入「資料集轉換工具」**：上傳圖像與標註
2. 🏋️ **前往「模型訓練」**：訓練 YOLO 模型
3. 🧮 **點選「NCNN量化」**：將模型轉為嵌入式部署格式
4. 📲 將量化模型部署至手機、樹莓派或其他端裝置
""")