from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


# TODO: Add group_identifier instead of hardcoding gauge_id
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
        Improved dataset that precomputes per-gauge tensors and a lightweight index.
        """
        self.input_length = input_length
        self.output_length = output_length
        self.total_length = input_length + output_length
        self.features = sorted(features)
        self.target = target
        self.static_features = sorted(static_features) if static_features else []

        # sort time series by gauge_id and date
        self.df_sorted = time_series_df.sort_values(["gauge_id", "date"])

        # Precompute static features per gauge (if provided)
        if static_df is not None:
            ts_gauge_ids = set(self.df_sorted["gauge_id"].unique())
            static_gauge_ids = set(static_df["gauge_id"].unique())
            missing = ts_gauge_ids - static_gauge_ids
            assert not missing, f"Missing static data for gauge ids: {missing}"
            self.static_dict = {
                row["gauge_id"]: torch.tensor(
                    row[self.static_features].to_numpy(dtype=np.float32)
                )
                for _, row in static_df.iterrows()
            }
        else:
            # If no static data, simply return zeros of appropriate size
            self.static_dict = {}

        # For each gauge, convert the time series data to tensors
        self.gauge_ids = []
        self.features_data: Dict[str, torch.Tensor] = {}
        self.target_data: Dict[str, torch.Tensor] = {}
        for gauge_id, group in self.df_sorted.groupby("gauge_id"):
            feat_tensor = torch.tensor(group[self.features].to_numpy(dtype=np.float32))
            targ_tensor = torch.tensor(group[self.target].to_numpy(dtype=np.float32))
            self.features_data[gauge_id] = feat_tensor
            self.target_data[gauge_id] = targ_tensor
            self.gauge_ids.append(gauge_id)

        # Build an index DataFrame with one row per valid sequence.
        index_list = []
        for gauge_id in self.gauge_ids:
            feat_tensor = self.features_data[gauge_id]
            n_steps = feat_tensor.shape[0]
            # For each gauge, every possible window of total_length is a candidate.
            # We only store the start index if the target window has no NaNs.
            for start in range(n_steps - self.total_length + 1):
                targ_window = self.target_data[gauge_id][
                    start + self.input_length : start + self.total_length
                ]
                if not torch.isnan(targ_window).any():
                    index_list.append((gauge_id, start))
        self.index = pd.DataFrame(index_list, columns=["gauge_id", "start_idx"])
        print(f"Created {len(self.index)} valid sequences.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Retrieve the gauge and starting index for this sequence from the index table.
        row = self.index.iloc[idx]
        gauge_id = row["gauge_id"]
        start_idx = int(row["start_idx"])
        end_idx = start_idx + self.total_length

        # Slice the precomputed tensors
        X = self.features_data[gauge_id][start_idx : start_idx + self.input_length]
        y = self.target_data[gauge_id][start_idx + self.input_length : end_idx]

        # Get static features (if any)
        if self.static_dict:
            static = self.static_dict.get(
                gauge_id, torch.zeros(len(self.static_features), dtype=torch.float32)
            )
        else:
            static = torch.zeros(0)

        return {
            "X": X,  # shape: (input_length, num_features)
            "y": y,  # shape: (output_length,)
            "static": static,  # shape: (len(static_features),)
            "gauge_id": gauge_id,
        }
