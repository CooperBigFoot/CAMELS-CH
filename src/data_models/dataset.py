from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


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
        group_identifier: str = "gauge_id",
    ) -> None:
        """
        Dataset class for hydrological data that handles both time series and static features.

        Args:
            time_series_df: DataFrame with time series data
            input_length: Number of input time steps
            output_length: Number of output time steps
            features: List of feature column names
            target: Target column name
            static_df: Optional DataFrame with static features
            static_features: Optional list of static feature names
            group_identifier: Column name for basin/gauge identifier
        """
        self.input_length = input_length
        self.output_length = output_length
        self.total_length = input_length + output_length
        self.features = sorted(features)
        self.target = target
        self.group_identifier = group_identifier

        # Remove group_identifier from static_features if present
        if static_features:
            self.static_features = sorted(
                [f for f in static_features if f != group_identifier]
            )
        else:
            self.static_features = []

        # Sort time series by gauge_id and date
        self.df_sorted = time_series_df.sort_values([self.group_identifier, "date"])

        # Precompute static features per gauge (if provided)
        if static_df is not None and self.static_features:
            ts_gauge_ids = set(self.df_sorted[self.group_identifier].unique())
            static_gauge_ids = set(static_df[self.group_identifier].unique())
            missing = ts_gauge_ids - static_gauge_ids
            assert not missing, f"Missing static data for gauge ids: {missing}"

            # Create dictionary of static feature tensors per gauge
            self.static_dict = {
                row[self.group_identifier]: torch.tensor(
                    row[self.static_features].to_numpy(dtype=np.float32)
                )
                for _, row in static_df.iterrows()
            }
        else:
            # If no static data, return empty tensor
            self.static_dict = {}

        # For each gauge, convert the time series data to tensors
        self.gauge_ids = []
        self.features_data: Dict[str, torch.Tensor] = {}
        self.target_data: Dict[str, torch.Tensor] = {}
        for gauge_id, group in self.df_sorted.groupby(self.group_identifier):
            feat_tensor = torch.tensor(group[self.features].to_numpy(dtype=np.float32))
            targ_tensor = torch.tensor(group[self.target].to_numpy(dtype=np.float32))
            self.features_data[gauge_id] = feat_tensor
            self.target_data[gauge_id] = targ_tensor
            self.gauge_ids.append(gauge_id)

        # Build an index DataFrame with one row per valid sequence
        index_list = []
        for gauge_id in self.gauge_ids:
            feat_tensor = self.features_data[gauge_id]
            n_steps = feat_tensor.shape[0]
            for start in range(n_steps - self.total_length + 1):
                targ_window = self.target_data[gauge_id][
                    start + self.input_length : start + self.total_length
                ]
                if not torch.isnan(targ_window).any():
                    index_list.append((gauge_id, start))

        self.index = pd.DataFrame(
            index_list, columns=[self.group_identifier, "start_idx"]
        )
        print(f"Created {len(self.index)} valid sequences")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Retrieve the gauge and starting index for this sequence
        row = self.index.iloc[idx]
        gauge_id = row[self.group_identifier]
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
            static = torch.zeros(0, dtype=torch.float32)

        return {
            "X": X,  # shape: (input_length, num_features)
            "y": y,  # shape: (output_length,)
            "static": static,  # shape: (len(static_features),)
            self.group_identifier: gauge_id,
        }
