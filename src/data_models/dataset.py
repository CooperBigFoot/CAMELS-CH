from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional


class HydroDataset(Dataset):
    def __init__(
        self,
        time_series_df: pd.DataFrame,
        input_length: int,
        output_length: int,
        features: List[str],
        target: str,
        static_df: Optional[pd.DataFrame] = None,
        static_features: Optional[List[str]] = None,
    ) -> None:
        """
        Create a PyTorch dataset with precomputed numpy arrays for efficiency.

        Args:
            time_series_df: DataFrame containing the time series data.
            static_df: DataFrame containing the static data.
            input_length: Number of time steps to use as input.
            output_length: Number of time steps to predict.
            features: List of features to use as input.
            target: Target variable to predict.
            static_features: List of static features to use.
        """
        self.input_length = input_length
        self.output_length = output_length
        self.features = sorted(features)
        self.target = target
        self.static_features = sorted(static_features) if static_features else []

        # Sort time series data by gauge_id and date
        self.df_sorted = time_series_df.sort_values(["gauge_id", "date"])

        if static_df is not None:
            ts_gauge_ids = set(self.df_sorted["gauge_id"].unique())
            static_gauge_ids = set(static_df["gauge_id"].unique())
            missing = ts_gauge_ids - static_gauge_ids
            assert not missing, f"Missing static data for gauge ids: {missing}"

            self.static_features_dict = {
                row["gauge_id"]: row[self.static_features].to_numpy(dtype=np.float32)
                for _, row in static_df.iterrows()
            }
        else:
            self.static_features_dict = None

        # Precompute numpy arrays for time series data per gauge_id
        self.timeseries_data = {}
        for gauge_id, group in self.df_sorted.groupby("gauge_id"):
            features_array = group[self.features].to_numpy(dtype=np.float32)
            target_array = group[self.target].to_numpy(dtype=np.float32)
            self.timeseries_data[gauge_id] = (features_array, target_array)

        # Precompute valid sequences
        self.sequences = []
        total_length = self.input_length + self.output_length
        for gauge_id, (features_array, target_array) in self.timeseries_data.items():
            n_steps = features_array.shape[0]
            for i in range(n_steps - total_length + 1):
                window_target = target_array[i + self.input_length : i + total_length]
                if not np.any(np.isnan(window_target)):
                    self.sequences.append(
                        {
                            "gauge_id": gauge_id,
                            "start_idx": i,
                            "end_idx": i + total_length,
                        }
                    )
        print(f"Created {len(self.sequences)} valid sequences.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        gauge_id = seq["gauge_id"]
        start_idx = seq["start_idx"]
        end_idx = seq["end_idx"]
        features_array, target_array = self.timeseries_data[gauge_id]

        X = features_array[start_idx : start_idx + self.input_length]
        y = target_array[start_idx + self.input_length : end_idx]

        if self.static_features_dict is not None:
            static = self.static_features_dict.get(
                gauge_id, np.zeros(len(self.static_features), dtype=np.float32)
            )
        else:
            static = np.zeros(len(self.static_features), dtype=np.float32)

        return {
            "X": torch.from_numpy(X),
            "y": torch.from_numpy(y),
            "static": torch.from_numpy(static),
            "gauge_id": gauge_id,
        }
