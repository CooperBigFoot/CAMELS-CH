from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union


class HydroDataset(Dataset):
    """Dataset class for hydrological data with optional domain support.

    Handles time series and static features while providing domain identification
    capabilities for transfer learning scenarios. When domain_id is not specified,
    defaults to a standard training configuration.
    """

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
        domain_id: str = "source",
    ) -> None:
        """Initialize the dataset with optional domain awareness.

        Args:
            time_series_df: DataFrame containing time series data
            input_length: Length of input sequences
            output_length: Length of output sequences (prediction horizon)
            features: List of feature names to use
            target: Name of target variable
            static_df: Optional DataFrame containing static catchment attributes
            static_features: Optional list of static feature names to use
            group_identifier: Column name identifying the grouping variable
            domain_id: Identifier for the domain (default is "source")
        """
        self.input_length = input_length
        self.output_length = output_length
        self.total_length = input_length + output_length
        self.features = sorted(features)  # Sort for consistency
        self.target = target
        self.group_identifier = group_identifier
        self.domain_id = domain_id  # Store domain identifier

        # Process static features
        if static_features:
            self.static_features = sorted(
                [f for f in static_features if f != group_identifier]
            )
        else:
            self.static_features = []

        # Sort time series data for consistency
        self.df_sorted = time_series_df.sort_values(
            [self.group_identifier, "date"])

        # Handle static features
        if static_df is not None and self.static_features:
            # Validate static data coverage
            ts_gauge_ids = set(self.df_sorted[self.group_identifier].unique())
            static_gauge_ids = set(static_df[self.group_identifier].unique())
            missing = ts_gauge_ids - static_gauge_ids
            if missing:
                raise ValueError(
                    f"Domain {domain_id}: Missing static data for gauge ids: {missing}"
                )

            # Create static feature dictionary
            self.static_dict = {
                row[self.group_identifier]: torch.tensor(
                    row[self.static_features].to_numpy(dtype=np.float32)
                )
                for _, row in static_df.iterrows()
            }
        else:
            self.static_dict = {}

        # Convert time series to tensors
        self.gauge_ids = []
        self.features_data: Dict[str, torch.Tensor] = {}
        self.target_data: Dict[str, torch.Tensor] = {}

        for gauge_id, group in self.df_sorted.groupby(self.group_identifier):
            # Convert features and target to tensors
            feat_tensor = torch.tensor(
                group[self.features].to_numpy(dtype=np.float32))
            targ_tensor = torch.tensor(
                group[self.target].to_numpy(dtype=np.float32))

            self.features_data[gauge_id] = feat_tensor
            self.target_data[gauge_id] = targ_tensor
            self.gauge_ids.append(gauge_id)

        # Build index of valid sequences
        self._build_sequence_index()

        print(
            f"Domain {domain_id}: Created {len(self.index)} valid sequences "
            f"from {len(self.gauge_ids)} catchments"
        )

    def _build_sequence_index(self) -> None:
        """Build index of valid sequences, excluding those with NaN values."""
        index_list = []

        for gauge_id in self.gauge_ids:
            feat_tensor = self.features_data[gauge_id]
            targ_tensor = self.target_data[gauge_id]
            n_steps = feat_tensor.shape[0]

            # Find valid sequences
            for start in range(n_steps - self.total_length + 1):
                feat_window = feat_tensor[start: start + self.input_length]
                targ_window = targ_tensor[
                    start + self.input_length: start + self.total_length
                ]

                # Only include sequences without NaN values
                if (
                    not torch.isnan(feat_window).any()
                    and not torch.isnan(targ_window).any()
                ):
                    index_list.append((gauge_id, start))

        self.index = pd.DataFrame(
            index_list, columns=[self.group_identifier, "start_idx"]
        )

    def __len__(self) -> int:
        """Return number of valid sequences in the dataset."""
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sequence with domain information.

        Args:
            idx: Index of the sequence to retrieve

        Returns:
            Dictionary containing:
                - X: Input features tensor
                - y: Target tensor
                - static: Static features tensor
                - domain_id: Domain identifier (if not "training")
                - group_identifier: Basin/gauge identifier
        """
        # Get sequence information
        row = self.index.iloc[idx]
        gauge_id = row[self.group_identifier]
        start_idx = int(row["start_idx"])
        end_idx = start_idx + self.total_length

        # Extract sequence data
        X = self.features_data[gauge_id][start_idx: start_idx +
                                         self.input_length]
        y = self.target_data[gauge_id][start_idx + self.input_length: end_idx]

        # Get static features or zeros if none available
        static = (
            self.static_dict.get(
                gauge_id, torch.zeros(
                    len(self.static_features), dtype=torch.float32)
            )
            if self.static_dict
            else torch.zeros(0, dtype=torch.float32)
        )

        # Build return dictionary
        domain_tensor = torch.tensor([1.0 if self.domain_id == "target" else 0.0],
                                     dtype=torch.float32)

        return_dict = {
            "X": X,
            "y": y,
            "static": static,
            "domain_id": domain_tensor,
            self.group_identifier: gauge_id,
        }

        return return_dict

