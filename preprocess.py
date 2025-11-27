import pandas as pd
import numpy as np

def engineer_features(df):
    grouped = df.groupby(['OEM', 'cell'])
    records = []

    for (oem, cell), group in grouped:
        discharge = group[group['current'] < 0]
        soh = abs(discharge['capacity'].max()) / 2.0  # assume rated = 2Ah
        sop = abs(discharge['power'].sum()) / 8.0     # assume rated = 8W

        record = {
            'OEM': oem,
            'cell': cell,
            'soh': soh,
            'sop': sop,
            'v_mean': discharge['voltage'].mean(),
            'i_peak': discharge['current'].min(),
            'energy': discharge['feature10'].max()
        }
        records.append(record)

    return pd.DataFrame(records)

def main():
    df = pd.read_csv("data/processed/full_data.csv")
    features_df = engineer_features(df)
    features_df.to_csv("data/processed/features.csv", index=False)
    print("Saved: data/processed/features.csv")

if __name__ == "__main__":
    main()
