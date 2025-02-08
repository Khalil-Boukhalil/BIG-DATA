import pandas as pd
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["parking_db"]
collection = db["yolo_predictions"]

data = list(collection.find({}, {"_id": 0, "timestamp": 1, "free_spaces": 1}))

df = pd.DataFrame(data)

df['timestamp'] = pd.to_datetime(df['timestamp'])

df = df.sort_values(by="timestamp")

df.to_csv("parking_data.csv", index=False)

print("✅ Parking data extracted and saved to parking_data.csv")
