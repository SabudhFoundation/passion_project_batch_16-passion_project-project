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
    return pd.concat(dfs, ignore_index=True, sort=False)

def sort_dataframe(df):
    return df.sort_values(
        by=["District Name", "Block Name", "Panchayat Name"]
    )

def save_output(df, filename):
    df.to_csv(filename, index=False)

def main():
    file_pattern = "mgnrega_Demographics_*_Fast.csv"
    
    files = get_csv_files(file_pattern)

    if not files:
        print("❌ No files found!")
        return

    dfs = read_csv_files(files)

    combined_df = combine_dataframes(dfs)

    # ✅ Sorting
    combined_df = sort_dataframe(combined_df)

    save_output(combined_df, "merged_sorted_mgnrega_data.csv")

    print(f"✅ Successfully merged {len(files)} files!")
    print("📁 Output: merged_sorted_mgnrega_data.csv")

if __name__ == "__main__":
    main()