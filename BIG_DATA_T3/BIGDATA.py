import os
import torch
import json
import cv2
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

DATASET_PATH = r"C:\Users\khali\PycharmProjects\BIG_DATA_T3\PKlot"
DATASET_YAML = os.path.join(DATASET_PATH, "data.yaml")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("🚨 Dataset folder not found!")

if not os.path.exists(DATASET_YAML):
    raise FileNotFoundError("🚨 dataset.yaml file is missing!")

print("✅ Dataset and YAML file verified!")


def clean_dataset(dataset_path):
    image_dir = os.path.join(dataset_path, "train", "images")
    label_dir = os.path.join(dataset_path, "train", "labels")

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        raise FileNotFoundError("🚨 Image or label directory missing!")

    image_files = set(f for f in os.listdir(image_dir) if f.endswith(".jpg"))
    label_files = set(f for f in os.listdir(label_dir) if f.endswith(".txt"))

    images_without_labels = [img for img in image_files if img.replace(".jpg", ".txt") not in label_files]
    labels_without_images = [lbl for lbl in label_files if lbl.replace(".txt", ".jpg") not in image_files]

    for img in images_without_labels:
        os.remove(os.path.join(image_dir, img))

    for lbl in labels_without_images:
        os.remove(os.path.join(label_dir, lbl))

    print(f"✅ Cleaned dataset: {len(images_without_labels)} images & {len(labels_without_images)} labels removed.")


clean_dataset(DATASET_PATH)

MODEL_NAME = "yolov8n.pt"
model = YOLO(MODEL_NAME)


def show_sample_image(dataset_path):
    """Displays a sample image with YOLO bounding boxes."""
    image_dir = os.path.join(dataset_path, "train", "images")
    label_dir = os.path.join(dataset_path, "train", "labels")

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    if not image_files:
        print("❌ No images found in dataset.")
        return

    sample_image = image_files[0]
    sample_image_path = os.path.join(image_dir, sample_image)
    sample_label_path = os.path.join(label_dir, sample_image.replace(".jpg", ".txt"))

    image = cv2.imread(sample_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if os.path.exists(sample_label_path):
        with open(sample_label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id, x, y, w, h = map(float, parts)
                h_img, w_img, _ = image.shape
                x, y, w, h = int(x * w_img), int(y * h_img), int(w * w_img), int(h * h_img)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    plt.imshow(image)
    plt.axis("off")
    plt.title("Sample Image with Bounding Box")
    plt.show()


show_sample_image(DATASET_PATH)


def train_yolo():
    print("🚀 Starting YOLO Training...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    model.train(
        data=DATASET_YAML,
        epochs=20,
        imgsz=640,
        batch=16,
        device=device,
        optimizer="AdamW",
        lr0=0.01,
        patience=5,
        augment=True,
        degrees=10,
        mixup=0.1,
        workers=0,
    )
    print("✅ Training Completed!")
if __name__ == "__main__":
    train_yolo()
