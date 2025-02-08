from kafka import KafkaConsumer
import json
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["parking_db"]
collection = db["yolo_predictions"]

consumer = KafkaConsumer(
    "parking_predictions",
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda v: json.loads(v.decode("utf-8"))
)

print("🚀 Listening for YOLO parking predictions...")

for message in consumer:
    parking_data = message.value
    print(f" Received YOLO prediction: {parking_data}")

    collection.insert_one(parking_data)
    print(" YOLO prediction data inserted into MongoDB!")
