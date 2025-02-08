from flask import Flask, render_template, jsonify, send_from_directory
from pymongo import MongoClient
import os
from datetime import datetime

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["parking_db"]
collection = db["yolo_predictions"]

# Path to the folder containing images
IMAGE_FOLDER = "C:/Users/khali/PycharmProjects/BIG_DATA_T3/parking_images"


@app.route('/')
def index():
    latest_data = list(collection.find().sort("timestamp", -1).limit(5000))  # Get latest 30 records

    parking_status = []
    for data in latest_data:
        # Convert timestamp to a more readable format
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)  # In case it's stored as a string

        # Format the timestamp to something more readable
        formatted_timestamp = timestamp.strftime("%B %d, %Y, %I:%M %p")  # Example format: February 8, 2025, 1:59 PM

        status = "Available" if data["free_spaces"] > 0 else "Full"
        parking_status.append({
            "lot_id": data["lot_id"],
            "detected_cars": data["detected_cars"],
            "total_spaces": data["total_spaces"],
            "free_spaces": data["free_spaces"],
            "status": status,
            "timestamp": formatted_timestamp,  # Pass the formatted timestamp
            "image_path": data.get("image_path", "")  # Add image path to data
        })

    return render_template("index.html", parking_status=parking_status, image_folder=IMAGE_FOLDER)


@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


@app.route('/api/parking_status')
def api_parking_status():
    latest_data = list(collection.find().sort("timestamp", -1).limit(5000))  # Get latest 30 records

    parking_status = []
    for data in latest_data:
        # Convert timestamp to a more readable format
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)  # In case it's stored as a string

        # Format the timestamp to something more readable
        formatted_timestamp = timestamp.strftime("%B %d, %Y, %I:%M %p")  # Example format: February 8, 2025, 1:59 PM

        status = "Available" if data["free_spaces"] > 0 else "Full"
        parking_status.append({
            "lot_id": data["lot_id"],
            "detected_cars": data["detected_cars"],
            "total_spaces": data["total_spaces"],
            "free_spaces": data["free_spaces"],
            "status": status,
            "timestamp": formatted_timestamp,  # Pass the formatted timestamp
            "image_path": data.get("image_path", "")
        })

    return jsonify(parking_status)


if __name__ == '__main__':
    app.run(debug=True)
