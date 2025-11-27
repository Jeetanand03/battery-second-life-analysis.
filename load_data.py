import pandas as pd
import glob
import os

def load_oem_data(oem_name, path):
    files = glob.glob(f"{path}/{oem_name}/*.txt")
    df_list = []
    for file in files:
        df = pd.read_csv(file, delim_whitespace=True, header=None)
        df.columns = ['id1', 'cell', 'step', 'id2', 'count', 'time',
                      'time_min', 'voltage', 'current', 'capacity',
                      'feature10', 'power']
        df = df.drop(columns=['id1', 'id2'])
        df['OEM'] = oem_name
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def main():
    raw_path = "data/raw"
    all_data = pd.concat([
        load_oem_data("OEM_A", raw_path),
        load_oem_data("OEM_B", raw_path),
        load_oem_data("OEM_C", raw_path)
    ])
    os.makedirs("data/processed", exist_ok=True)
    all_data.to_csv("data/processed/full_data.csv", index=False)
    print("Saved: data/processed/full_data.csv")

if __name__ == "__main__":
    main()
