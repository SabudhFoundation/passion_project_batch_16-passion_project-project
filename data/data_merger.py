import pandas as pd
import re
from rapidfuzz import process, fuzz

def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r'\s+', ' ', str(text).lower().strip())

def clean_block_name(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Removes ( 226 ) or [226] from names to match JJM/MGNREGA blocks
    text = re.sub(r'\s*[\(\[]\s*\d+\s*[\)\]]\s*', '', text)
    return clean_text(text)

def apply_fuzzy_matching(base_df, target_df, block_col, name_col, threshold=85):
    """Fuzzy matches panchayat names block-by-block using substring-friendly logic."""
    master_choices_per_block = {
        k: list(v) for k, v in base_df.groupby(block_col)[name_col].unique().items()
    }

    def find_best_match(row):
        block = row[block_col]
        original_name = row[name_col]
        
        choices = master_choices_per_block.get(block, [])
        if not choices or not isinstance(original_name, str) or original_name == "":
            return original_name
            
        if original_name in choices:
            return original_name
            
        # Using token_set_ratio for better substring matching
        best_match = process.extractOne(original_name, choices, scorer=fuzz.token_set_ratio)
        
        if best_match and best_match[1] >= threshold:
            return best_match[0]
            
        return original_name

    target_df[name_col] = target_df.apply(find_best_match, axis=1)
    return target_df

def generate_final_csv():
    print("Loading 4 datasets...")
    df_sbm = pd.read_csv("sbm_data.csv")
    df_pai = pd.read_csv("PAI_Village_Data.csv")
    df_jjm = pd.read_csv("jjm_data.csv")
    df_mgnrega = pd.read_csv("merged_sorted_mgnrega_data.csv")

    # --- Step 1: Prep Group 1 (SBM + PAI) ---
    print("Processing SBM and PAI...")
    df_sbm['GP_ID'] = df_sbm['GP_ID'].astype(str).str.strip()
    df_pai['GP_ID'] = df_pai['GP_ID'].astype(str).str.strip()
    df_pai = df_pai.drop_duplicates(subset=['GP_ID'])
    
    df_final = pd.merge(df_sbm, df_pai, on='GP_ID', how='left')
    
    # Standardize keys for the fuzzy merge later
    df_final['panchayat_clean'] = df_final['GP_NAME'].apply(clean_text)
    df_final['block_clean'] = df_final['BLOCK NAME'].apply(clean_block_name)

    # --- Step 2: Prep Group 2 (JJM + MGNREGA) ---
    print("Processing JJM and MGNREGA...")
    df_jjm['panchayat_clean'] = df_jjm['PANCHAYAT_NAME'].apply(clean_text)
    df_jjm['block_clean'] = df_jjm['BLOCK_NAME'].apply(clean_text)

    df_mgnrega['panchayat_clean'] = df_mgnrega['Panchayat Name'].apply(clean_text)
    df_mgnrega['block_clean'] = df_mgnrega['Block Name'].apply(clean_text)
    df_mgnrega = df_mgnrega.drop_duplicates(subset=['block_clean', 'panchayat_clean'])

    # Align JJM spellings to MGNREGA spellings
    df_jjm = apply_fuzzy_matching(df_mgnrega, df_jjm, 'block_clean', 'panchayat_clean', threshold=85)
    df_group2 = pd.merge(df_jjm, df_mgnrega, on=['block_clean', 'panchayat_clean'], how='left')

    # Since MGNREGA/JJM are at Panchayat level here, we drop duplicates to merge into SBM villages
    df_group2_unique = df_group2.drop_duplicates(subset=['block_clean', 'panchayat_clean'])

    # --- Step 3: Final Merge (Group 1 + Group 2) ---
    print("Performing final fuzzy alignment and merging...")
    # Align the JJM/MGNREGA names to match SBM/PAI naming conventions
    df_group2_unique = apply_fuzzy_matching(
        base_df=df_final, 
        target_df=df_group2_unique.copy(), 
        block_col='block_clean', 
        name_col='panchayat_clean', 
        threshold=85
    )

    # Final Merge
    df_master = pd.merge(df_final, df_group2_unique, on=['block_clean', 'panchayat_clean'], how='left')

    # Clean up: Remove the temporary 'clean' columns before saving
    df_master = df_master.drop(columns=['panchayat_clean', 'block_clean'])

    # Save to CSV
    output_file = "final_merged_data.csv"
    df_master.to_csv(output_file, index=False)
    
    print(f"\n--- Merge Complete ---")
    print(f"Total Villages: {len(df_master)}")
    print(f"Data saved to: {output_file}")

if __name__ == "__main__":
    generate_final_csv()