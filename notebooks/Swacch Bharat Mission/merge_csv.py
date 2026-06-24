import pandas as pd
import glob
import re

def get_csv_files(pattern):
    return glob.glob(pattern)

def clean_column_names(df):
    # Standardize column names (remove spaces issues, uppercase)
    df.columns = [col.strip().upper() for col in df.columns]
    return df

def split_name_id(series):
    names = []
    ids = []
    
    for value in series.astype(str):
        match = re.search(r"(.*?)\s*\(\s*(\d+)\s*\)", value)
        if match:
            names.append(match.group(1).strip())
            ids.append(match.group(2))
        else:
            names.append(value.strip())
            ids.append(None)
    
    return names, ids

def process_dataframe(df):
    df = clean_column_names(df)

    # Split Panchayat
    if "PANCHAYAT NAME" in df.columns:
        gp_names, gp_ids = split_name_id(df["PANCHAYAT NAME"])
        df["GP_NAME"] = gp_names
        df["GP_ID"] = gp_ids

    # Split Village
    if "VILLAGE NAME" in df.columns:
        v_names, v_ids = split_name_id(df["VILLAGE NAME"])
        df["VILLAGE_NAME"] = v_names
        df["VILLAGE_ID"] = v_ids

    return df

def clean_and_reorder(df):
    # Remove original columns
    df = df.drop(columns=["VILLAGE NAME", "PANCHAYAT NAME"], errors="ignore")
    
    # Desired order
    priority_cols = ["VILLAGE_ID", "VILLAGE_NAME", "GP_ID", "GP_NAME"]
    priority_cols = [col for col in priority_cols if col in df.columns]
    
    remaining_cols = [col for col in df.columns if col not in priority_cols]
    
    return df[priority_cols + remaining_cols]

def sort_dataframe(df):
    sort_cols = ["DISTRICT NAME", "BLOCK NAME", "GP_NAME", "VILLAGE_NAME"]
    sort_cols = [col for col in sort_cols if col in df.columns]
    
    return df.sort_values(by=sort_cols)

def save_output(df, filename):
    df.to_csv(filename, index=False)

def main():
    file_pattern = "sbm_data_*.csv"
    
    files = get_csv_files(file_pattern)

    if not files:
        print("❌ No CSV files found!")
        return

    dataframes = []
    
    for file in files:
        df = pd.read_csv(file)
        df = process_dataframe(df)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df = sort_dataframe(combined_df)
    combined_df = clean_and_reorder(combined_df)

    save_output(combined_df, "final_cleaned_sbm_data.csv")

    print(f"✅ Successfully processed {len(files)} files!")
    print("📁 Output file: final_cleaned_sbm_data.csv")

if __name__ == "__main__":
    main()