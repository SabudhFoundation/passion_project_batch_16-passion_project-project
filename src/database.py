from pymongo import MongoClient
import certifi  # <-- Add this import
from config import settings

# Tell MongoClient to use certifi's certificates for the SSL connection
client = MongoClient(settings.MONGO_URI, tlsCAFile=certifi.where()) 
db = client['VillageAnalytics']
collection = db['LSDG_Metrics']

def get_village_by_name(village_name: str) -> dict:
    """Fetches a specific village by exact or regex match."""
    if not village_name:
        return None
    query = {"village_name": {"$regex": f"^{village_name.strip()}$", "$options": "i"}}
    return collection.find_one(query)

def get_all_villages_list() -> list:
    """Returns a simple list of all village names for fuzzy matching/dropdowns."""
    cursor = collection.find({}, {"village_name": 1, "_id": 0})
    return [doc.get("village_name") for doc in cursor if doc.get("village_name")]

def search_villages_for_grid(village_name: str) -> list:
    """Searches for partial matches and returns basic info for the AgGrid."""
    if not village_name:
        return []
    # Partial match regex (e.g., searching "bal" will find "Baluana")
    query = {"village_name": {"$regex": village_name.strip(), "$options": "i"}}
    
    # Return only the fields needed for the table to save memory
    cursor = collection.find(query, {"_id": 0, "village_name": 1, "gp_name": 1, "block_name": 1})
    return list(cursor)