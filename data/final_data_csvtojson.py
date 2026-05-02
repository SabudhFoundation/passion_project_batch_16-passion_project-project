import pandas as pd
import json
import numpy as np

def convert_csv_to_nested_json(input_csv, output_json):
    # 1. Load the structured CSV
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)

    # 2. Identify column groups based on our naming convention
    id_cols = ['village_id', 'village_name', 'gp_id', 'gp_name', 'block_name', 'district_name', 'state_name']
    sanit_cols = [c for c in df.columns if c.startswith('sanit_')]
    pai_cols = [c for c in df.columns if c.startswith('pai_')]
    water_cols = [c for c in df.columns if c.startswith('water_')]
    mgnrega_cols = [c for c in df.columns if c.startswith('mgnrega_')]

    json_list = []

    print("Converting rows to nested objects...")
    for _, row in df.iterrows():
        # Create the base document with identity fields
        village_doc = {}
        for col in id_cols:
            val = row[col]
            # Handle NaN values for JSON compatibility (converts to null)
            village_doc[col] = val if pd.notna(val) else None
            
        # Nest Sanitation metrics (removing 'sanit_' prefix)
        village_doc['sanitation'] = {
            c.replace('sanit_', ''): (row[c] if pd.notna(row[c]) else 0) 
            for c in sanit_cols
        }
        
        # Nest Governance metrics (removing 'pai_' prefix)
        village_doc['governance'] = {
            c.replace('pai_', ''): (row[c] if pd.notna(row[c]) else None) 
            for c in pai_cols
        }
        
        # Nest Water Security metrics (removing 'water_' prefix)
        village_doc['water_security'] = {
            c.replace('water_', ''): (row[c] if pd.notna(row[c]) else None) 
            for c in water_cols
        }
        
        # Nest Employment metrics (removing 'mgnrega_' prefix)
        village_doc['employment'] = {
            c.replace('mgnrega_', ''): (row[c] if pd.notna(row[c]) else 0) 
            for c in mgnrega_cols
        }
        
        json_list.append(village_doc)

    # 3. Save to a JSON file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_list, f, indent=4)
    
    print(f"Successfully created {output_json} with {len(json_list)} records.")

if __name__ == "__main__":
    convert_csv_to_nested_json('final_structured_village_data.csv', 'final_village_data.json')