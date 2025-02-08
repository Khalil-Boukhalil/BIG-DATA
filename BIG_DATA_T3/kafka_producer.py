from kafka import KafkaProducer
import json
import time
from datetime import datetime, timezone

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

parking_updates = [
    {"lot_id": "P1", "location": "Downtown", "total_spaces": 100, "occupied_spaces": 35, "timestamp": datetime.now(timezone.utc).isoformat()},
    {"lot_id": "P2", "location": "Airport Zone", "total_spaces": 200, "occupied_spaces": 55, "timestamp": datetime.now(timezone.utc).isoformat()}
]

for update in parking_updates:
    producer.send("parking_topic", update)
    print(f"🚀 Sent update: {update}")
    time.sleep(2)

producer.flush()
producer.close()
