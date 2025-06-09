import cv2
import os

# 輸入資料夾與輸出標註資料夾
image_folder = "images"             # 原始圖片資料夾
label_folder = "labels"             # 儲存標註的 .txt 檔案
os.makedirs(label_folder, exist_ok=True)

# 請根據你的需求設定類別（目前預設只有一類 class 0）
class_id = 0

# 處理每張圖片
for filename in os.listdir(image_folder):
    if not filename.endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_lines = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        # 忽略太小的框
        if bw * bh < 100:
            continue

        # 轉換成 YOLO 格式：中心點 + 寬高，並歸一化
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        bw_norm = bw / w
        bh_norm = bh / h

        label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}")

    # 儲存為 YOLO 標註檔
    base_name = os.path.splitext(filename)[0]
    with open(os.path.join(label_folder, base_name + ".txt"), "w") as f:
        f.write("\n".join(label_lines))
