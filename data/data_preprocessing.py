import pandas as pd
import numpy as np

def clean_and_structure_data(input_file, output_file):
    # 1. Load the merged data
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    # 2. Define the Mapping Logic
    # This keeps only 1 set of ID columns and prefixes others for clarity
    id_map = {
        'VILLAGE_ID': 'village_id',
        'VILLAGE_NAME_x': 'village_name',
        'GP_ID': 'gp_id',
        'GP_NAME': 'gp_name',
        'BLOCK NAME': 'block_name',
        'DISTRICT NAME': 'district_name',
        'STATE NAME': 'state_name'
    }

    # Define thematic prefixes
    mapping = id_map.copy()
    for col in df.columns:
        col_clean = col.strip()
        
        # Sanitation Metrics (SBM)
        if 'TOTAL -' in col or 'ODF' in col:
            new_name = 'sanit_' + col_clean.replace('TOTAL - ', '').replace(' ', '_').lower()
            mapping[col] = new_name
            
        # Governance Metrics (PAI)
        elif any(x in col for x in ['PAI_Score', 'Grade', 'Category', '_Score']):
            mapping[col] = 'pai_' + col_clean.lower()
            
        # Water Metrics (JJM)
        elif col in ['TOTAL_POPULATION', 'SC_POPULATION', 'ST_POPULATION', 'GEN_POPULATION', 
                    'NO._OF_HABITATIONS', 'IS_PWS_AVAILABLE', 'JJM_STATUS', 
                    'HAR_GHAR_JAL_CERTIFICATE', 'TOTAL_HOUSEHOLDS', 'TOTAL_TAP_CONNECTIONS']:
            mapping[col] = 'water_' + col_clean.lower().replace('.', '')
            
        # Employment Metrics (MGNREGA)
        elif col in ['Total Registered HH', 'Total Registered Persons', 'SC HH', 'SC Persons', 
                    'ST HH', 'ST Persons', 'Male Persons', 'Female Persons']:
            mapping[col] = 'mgnrega_' + col_clean.replace(' ', '_').lower()

    # 3. Filter and Rename
    # This automatically removes the redundant columns (like 'District Name_y') 
    # because they are not in our 'mapping' dictionary.
    df_final = df[list(mapping.keys())].rename(columns=mapping)

    # 4. Deep Cleaning
    print("Standardizing text and handling nulls...")
    for col in df_final.select_dtypes(include=['object']).columns:
        # Convert 'None' strings and empty spaces to actual NaN
        df_final[col] = df_final[col].astype(str).str.strip().replace(['None', 'none', 'nan', 'NaN', ''], np.nan)

    # 5. Quality Filter
    # Keep only rows that have the core metrics from all 4 datasets
    critical_cols = ['pai_overall_pai_score', 'water_jjm_status', 'mgnrega_total_registered_hh']
    df_cleaned = df_final.dropna(subset=critical_cols)

    # 6. Fill zeros for specific numeric metrics
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(0)

    # 7. Save to CSV
    df_cleaned.to_csv(output_file, index=False)
    print(f"Success! Cleaned data saved to {output_file}")
    print(f"Final shape: {df_cleaned.shape}")

if __name__ == "__main__":
    clean_and_structure_data('final_merged_data.csv', 'final_structured_village_data.csv')