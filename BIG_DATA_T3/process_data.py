import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler

client = MongoClient("mongodb://localhost:27017/")
db = client["parking_db"]
collection = db["yolo_predictions"]

cursor = collection.find({}, {"timestamp": 1, "free_spaces": 1, "_id": 0})
data = list(cursor)

df = pd.DataFrame(data)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by="timestamp")

scaler = MinMaxScaler()
df["free_spaces_scaled"] = scaler.fit_transform(df[["free_spaces"]])

df.to_csv("processed_parking_data.csv", index=False)
print(" Data preprocessing complete. Saved to 'processed_parking_data.csv'.")
