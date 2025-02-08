import os
import json
import time
import cv2
from ultralytics import YOLO
from datetime import datetime, timezone
from kafka import KafkaProducer

model = YOLO(r"C:\Users\khali\runs\detect\train67\weights\best.pt")

image_folder = r"C:\Users\khali\PycharmProjects\BIG_DATA_T3\parking_images"


image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
if not image_files:
    raise Exception(" No images found in the directory!")

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print(f" Found {len(image_files)} images. Processing...")

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    print(f"🔍 Processing image: {image_path}")

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f" Warning: Could not read image {image_path}. Skipping...")
        continue

    results = model(image)

    detected_cars = len(results[0].boxes)

    parking_data = {
        "image_id": os.path.basename(image_path),
        "lot_id": "P1",
        "detected_cars": detected_cars,
        "total_spaces": 100,
        "free_spaces": max(0, 100 - detected_cars),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    producer.send("parking_predictions", parking_data)
    print(f" Sent: {parking_data}")

    time.sleep(0.5)

producer.flush()
producer.close()
print(" All images processed & sent to Kafka.")
