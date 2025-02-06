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
        Create a PyTorch dataset for the CAMELS dataset.

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
                row["gauge_id"]: row[self.static_features].values
                for _, row in static_df.iterrows()
            }
        else:
            self.static_features_dict = None

        # Precompute a dictionary mapping gauge_id to its time series data for faster indexing
        self.timeseries_dict = {
            gauge_id: self.df_sorted[
                self.df_sorted["gauge_id"] == gauge_id
            ].reset_index(drop=True)
            for gauge_id in self.df_sorted["gauge_id"].unique()
        }

        # Create valid sequences from the precomputed timeseries data
        self.sequences = self._create_sequences()
        print(f"Created {len(self.sequences)} valid sequences.")

    def _create_sequences(self) -> List[Dict[str, Union[str, int]]]:
        """
        Create sequences of input and output data for each gauge_id.
        Only sequences with no NaN values in the target output window are kept.
        """
        sequences = []
        total_length = self.input_length + self.output_length

        for gauge_id, ts_df in self.timeseries_dict.items():
            features_array = ts_df[self.features].values
            target_array = ts_df[self.target].values

            # Slide through the time series for valid windows
            for i in range(len(ts_df) - total_length + 1):
                target_window = target_array[i + self.input_length : i + total_length]
                if not np.any(np.isnan(target_window)):
                    sequences.append(
                        {
                            "gauge_id": gauge_id,
                            "start_idx": i,
                            "end_idx": i + total_length,
                        }
                    )
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        gauge_id = seq["gauge_id"]
        start_idx = seq["start_idx"]
        end_idx = seq["end_idx"]

        # Retrieve precomputed time series data for this gauge
        catchment_data = self.timeseries_dict[gauge_id].iloc[start_idx:end_idx]
        X = catchment_data[self.features].values[: self.input_length]
        y = catchment_data[self.target].values[self.input_length :]

        # Handle missing static features
        if self.static_features_dict is not None:
            static = np.array(
                self.static_features_dict.get(
                    gauge_id, np.zeros(len(self.static_features))
                ),
                dtype=np.float32,
            )
        else:
            static = np.zeros(len(self.static_features), dtype=np.float32)

        return {
            "X": torch.FloatTensor(X),
            "y": torch.FloatTensor(y),
            "static": torch.FloatTensor(static),
            "gauge_id": gauge_id,
        }
