from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["smart_parking"]

parking_lots = db.parking_lots.find()

print("\n Parking Lots Data:")
for lot in parking_lots:
    print(lot)
