import pandas as pd
import glob
import os


def combine_camels_data(
    folder_path: str, data_naming_convention: str, columns_to_keep: list
) -> pd.DataFrame:

    csv_files = glob.glob(os.path.join(folder_path, data_naming_convention))
    print(f"Found {len(csv_files)} files")
    print(csv_files)

    dfs = []

    for file in csv_files:
        gauge_id = file.split("_")[-1].replace(".csv", "")

        df = pd.read_csv(file, header=0)

        # Select only required columns
        df = df[columns_to_keep]

        # Add gauge_id column
        df["gauge_id"] = gauge_id
        df["time_idx"] = range(0, len(df))

        # Append to list
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by gauge_id and time_idx
    combined_df = combined_df.sort_values(by=["gauge_id", "time_idx"]).reset_index(
        drop=True
    )

    return combined_df
