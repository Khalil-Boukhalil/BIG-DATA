from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["smart_parking"]
collection = db["parking_lots"]

# Aggregate data to compute the average occupancy rate
pipeline = [
    {"$group": {"_id": "$lot_id", "avg_occupancy": {"$avg": "$occupied_spaces"}}}
]

results = collection.aggregate(pipeline)
for result in results:
    print(result)
