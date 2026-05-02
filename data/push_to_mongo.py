import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

user = os.getenv("MONGO_USER")
password = os.getenv("MONGO_PASS")

uri = f"mongodb+srv://{user}:{password}@cluster0.jqx7mfb.mongodb.net/"

client = MongoClient(uri)
db = client["VillageAnalytics"]
collection = db["LSDG_Metrics"]

with open("final_village_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, list):
    collection.insert_many(data)
else:
    collection.insert_one(data)

print("Data uploaded successfully")