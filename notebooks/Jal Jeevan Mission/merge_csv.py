import pandas as pd
import glob

def get_csv_files(pattern):
    return glob.glob(pattern)

def read_csv_files(file_list):
    dataframes = []
    
    for file in file_list:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    return dataframes

def combine_dataframes(dfs):
    # Handles different columns automatically
    return pd.concat(dfs, ignore_index=True, sort=False)

def clean_column_names(df):
    df.columns = [col.strip().upper().replace(" ", "_") for col in df.columns]
    return df

def save_output(df, filename):
    df.to_csv(filename, index=False)

def main():
    file_pattern = "jjm_data_*_District.csv"
    
    files = get_csv_files(file_pattern)

    if not files:
        print("❌ No CSV files found!")
        return

    dfs = read_csv_files(files)
    combined_df = combine_dataframes(dfs)

    combined_df = clean_column_names(combined_df)

    save_output(combined_df, "merged_jjm_data.csv")

    print(f"✅ Combined {len(files)} files successfully!")
    print("📁 Output: merged_jjm_data.csv")

if __name__ == "__main__":
    main()