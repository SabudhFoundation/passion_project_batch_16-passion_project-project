from pymongo import MongoClient
import certifi
from src.config import settings

# Initialize MongoDB Client with SSL certificate verification
client = MongoClient(settings.MONGO_URI, tlsCAFile=certifi.where()) 
db = client['VillageAnalytics']
collection = db['LSDG_Metrics']

def get_village_records(village_name: str) -> list:
    """
    Fetches all assessment records for a specific village.
    This allows the UI to let users choose between different Assessment Years.
    """
    if not village_name:
        return []
        
    # Case-insensitive exact match for the village name
    query = {"village_name": {"$regex": f"^{village_name.strip()}$", "$options": "i"}}
    cursor = collection.find(query)
    
    docs = []
    for doc in cursor:
        # Convert MongoDB ObjectId to string to prevent Streamlit/PyArrow serialization errors
        if doc and '_id' in doc:
            doc['_id'] = str(doc['_id'])
        docs.append(doc)
        
    return docs

def get_all_villages_list() -> list:
    """
    Returns a unique list of all village names available in the database.
    Used for fuzzy matching and search suggestions.
    """
    # Use distinct to avoid duplicates if multiple years exist for the same village
    return collection.distinct("village_name")

def search_villages_for_grid(village_name: str) -> list:
    """
    Searches for partial matches to populate the selection AgGrid.
    Now includes 'assessment_year' in the results so users can see 
    available data points immediately.
    """
    if not village_name:
        return []

    # --- NEW: SMART REGEX MATCHING ---
    search_term = village_name.strip()
    if len(search_term) <= 2:
        # If user types 1 or 2 letters, only match villages STARTING with those letters
        regex_pattern = f"^{search_term}"
    else:
        # For longer inputs, allow matching anywhere in the string
        regex_pattern = search_term
        
    query = {"village_name": {"$regex": regex_pattern, "$options": "i"}}
    
    # Return basic info plus assessment_year to help the user identify records
    projection = {
        "_id": 0, 
        "village_name": 1, 
        "gp_name": 1, 
        "block_name": 1, 
        "assessment_year": 1
    }
    
    cursor = collection.find(query, projection)
    return list(cursor)

def get_village_by_year(village_name: str, year: str) -> dict:
    """
    Fetches the specific document for a village and a chosen assessment year.
    Handles the Streamlit string vs MongoDB integer mismatch gracefully.
    """
    # Try to cast the Streamlit string year back to an integer
    year_int = None
    try:
        year_int = int(year)
    except ValueError:
        pass # If it's a string like "2024-25", leave it as None

    # Build a query that matches EITHER the string OR the integer
    year_conditions = [{"assessment_year": year}]
    if year_int is not None:
        year_conditions.append({"assessment_year": year_int})

    query = {
        "village_name": {"$regex": f"^{village_name.strip()}$", "$options": "i"},
        "$or": year_conditions
    }
    
    doc = collection.find_one(query)
    
    # Prevent PyArrow Streamlit crashes
    if doc and '_id' in doc:
        doc['_id'] = str(doc['_id'])
        
    return doc