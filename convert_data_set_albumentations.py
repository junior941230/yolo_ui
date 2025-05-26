import shutil
import random
from pathlib import Path
import cv2
import albumentations as A

random.seed(42)
source_dir = Path("temp_data")

# 設定資料增強策略
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
    A.GaussNoise(p=0.2),
    A.Rotate(limit=10, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def load_labels(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        raw_labels = [line.strip().split() for line in f if line.strip()]
    bboxes = []
    class_labels = []
    for label in raw_labels:
        cls_id, x, y, w, h = label
        bboxes.append([float(x), float(y), float(w), float(h)])
        class_labels.append(int(cls_id))
    return bboxes, class_labels


def augment_image_and_labels(name, img_src, label_src, dst_img_dir, dst_label_dir, augment=True,augment_times =4 ):
    image = cv2.imread(str(img_src))
    h, w = image.shape[:2]

    bboxes, class_labels = load_labels(label_src)

    # 增強張數
    augmentations = augment_times if augment else 1

    for i in range(augmentations):
        if augment:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_classes = augmented["class_labels"]
            name_suffix = f"{name}_aug{i}"
        else:
            aug_img = image
            aug_bboxes = bboxes
            aug_classes = class_labels
            name_suffix = name

        # 儲存圖片與標籤
        cv2.imwrite(str(dst_img_dir / f"{name_suffix}.jpg"), aug_img)
        with open(dst_label_dir / f"{name_suffix}.txt", "w", encoding="utf-8") as f:
            for cls, box in zip(aug_classes, aug_bboxes):
                f.write(f"{cls} {' '.join(f'{v:.6f}' for v in box)}\n")


def copy_files_val(file_list, img_dst, label_dst):
    for name in file_list:
        for ext in [".png", ".jpg"]:
            img_src = source_dir / f"{name}{ext}"
            if img_src.exists():
                shutil.copy(img_src, img_dst / img_src.name)
                break
        label_src = source_dir / f"{name}.txt"
        if label_src.exists():
            shutil.copy(label_src, label_dst / label_src.name)


def convert_data_set(output_name="yolo_train_data", augment=True,
                     augment_times=4, val_percent=None, val_list=None):
    output_dir = Path(f"datasets/{output_name}")
    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"

    for folder in [images_train, images_val, labels_train, labels_val]:
        folder.mkdir(parents=True, exist_ok=True)

    # ✅ 支援 png 與 jpg
    image_files = sorted(set(f.stem for f in source_dir.glob("*.png")) |
                         set(f.stem for f in source_dir.glob("*.jpg")))

    # 手動 vs 隨機選 val
    if val_list is not None:
        val_files = val_list
        train_files = [name for name in image_files if name not in val_list]
    else:
        random.shuffle(image_files)
        split_idx = int(len(image_files) * (1 - val_percent))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

    for name in train_files:
        for ext in [".png", ".jpg"]:
            img_src = source_dir / f"{name}{ext}"
            if img_src.exists():
                break
        label_src = source_dir / f"{name}.txt"
        if img_src.exists() and label_src.exists():
            augment_image_and_labels(
                name, img_src, label_src, images_train, labels_train, augment,augment_times)

    copy_files_val(val_files, images_val, labels_val)

    # classes.txt 轉為 data.yaml
    classes_path = source_dir / "classes.txt"
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names: [")
        f.write(", ".join(f"'{cls}'" for cls in class_names))
        f.write("]\n")

    train_sample_count = len(train_files) * (4 if augment else 1)
    print(
        f"✅ 完成！已建立 YOLOv8 資料集（train {'含增強' if augment else '無增強'}，val 無擴增）於 {output_dir}")
    print(f"📂 訓練樣本數：{train_sample_count}")
    print(f"📂 驗證樣本數：{len(val_files)}")
    return train_sample_count, len(val_files)
