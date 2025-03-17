import numpy as np
import pandas as pd
import copy
from typing import Dict, List, Tuple, Union, Optional
import torch
import pytorch_lightning as pl


class BaseEvaluator:
    """Base class for model evaluation with shared functionality."""

    def __init__(self, horizons: List[int] = None):
        """
        Initialize the base evaluator.

        Args:
            horizons: List of forecast horizons to evaluate (in days)
        """
        self.horizons = horizons or [1]
        self.models = {}
        self.datamodules = {}
        self.results = {}
        self.default_datamodule = None

    def register_model(
        self,
        name: str,
        model: pl.LightningModule,
        datamodule: Optional[pl.LightningDataModule] = None,
    ):
        """
        Register a new model with its specific datamodule.

        Args:
            name: Name identifier for the model
            model: PyTorch Lightning model to register
            datamodule: Optional datamodule specific to this model
        """
        self.models[name] = copy.deepcopy(model)
        if datamodule:
            self.datamodules[name] = datamodule
        elif name not in self.datamodules and self.default_datamodule is None:
            print(
                f"Warning: No datamodule provided for model '{name}' and no default datamodule available"
            )

    def _extract_data(self, test_results: Dict[str, torch.Tensor]) -> Dict:
        """
        Extract data from test results into standard format.

        Args:
            test_results: Dictionary containing model outputs and observations

        Returns:
            Dictionary with extracted data in standard format
        """
        basin_ids = np.array(test_results["basin_ids"])
        preds = test_results["predictions"].cpu().numpy()
        obs = test_results["observations"].cpu().numpy()

        print(
            f"Evaluating results with shape: preds={preds.shape}, obs={obs.shape}, basin_ids={basin_ids.shape}"
        )

        # Ensure pred and obs dimensions match
        if preds.shape != obs.shape:
            raise ValueError(
                f"Prediction shape {preds.shape} doesn't match observation shape {obs.shape}"
            )

        extracted_data = {
            "basin_ids": basin_ids,
            "predictions": preds,
            "observations": obs,
        }

        # Add dates if available
        if "input_end_date" in test_results:
            extracted_data["input_end_date"] = test_results["input_end_date"]

        return extracted_data

    def _prepare_evaluation_dataframe(
        self, test_results: Dict[str, torch.Tensor], datamodule=None
    ) -> pd.DataFrame:
        """
        Create a flattened dataframe with predictions, observations, and metadata.

        Args:
            test_results: Dictionary containing model outputs and observations
            datamodule: Specific datamodule to use for inverse transformations

        Returns:
            DataFrame with predictions, observations, basin IDs, horizons and dates
        """
        # Data extraction
        basin_ids = np.array(test_results["basin_ids"])
        preds = test_results["predictions"].cpu().numpy()
        obs = test_results["observations"].cpu().numpy()

        print(
            f"Evaluating results with shape: preds={preds.shape}, obs={obs.shape}, basin_ids={basin_ids.shape}"
        )

        # Ensure pred and obs dimensions match
        if preds.shape != obs.shape:
            raise ValueError(
                f"Prediction shape {preds.shape} doesn't match observation shape {obs.shape}"
            )

        # Create expanded basin IDs and horizons
        if preds.ndim == 2:  # [batch_size, pred_len]
            horizons_per_sample = preds.shape[1]

            # Handle horizon mismatch - don't modify self.horizons, use a local variable
            current_horizons = self.horizons

            if horizons_per_sample != len(current_horizons):
                print(
                    f"Warning: Model output has {horizons_per_sample} horizons but evaluator configured with {len(current_horizons)} horizons"
                )
                # Use the actual horizons from model output
                current_horizons = list(range(1, horizons_per_sample + 1))
                print(f"Using adjusted horizons: {current_horizons}")

            # Flatten predictions and observations
            preds_flat = preds.flatten()
            obs_flat = obs.flatten()

            # Repeat each basin ID for each horizon in the output
            basin_ids_expanded = np.repeat(basin_ids, horizons_per_sample)

            # Create repeated horizons array matching the model's output structure
            horizons_expanded = np.tile(current_horizons, len(basin_ids))

            # Verify all arrays have matching lengths
            assert (
                len(preds_flat)
                == len(obs_flat)
                == len(basin_ids_expanded)
                == len(horizons_expanded)
            ), (
                f"Array length mismatch: preds_flat={len(preds_flat)}, obs_flat={len(obs_flat)}, "
                f"basin_ids_expanded={len(basin_ids_expanded)}, horizons_expanded={len(horizons_expanded)}"
            )

            # Create dates if available
            if "input_end_date" in test_results:
                input_end_dates = test_results["input_end_date"]

                # Ensure input_end_dates matches basin_ids length
                if len(input_end_dates) != len(basin_ids):
                    print(
                        f"Warning: input_end_dates length ({len(input_end_dates)}) doesn't match basin_ids length ({len(basin_ids)})"
                    )
                    # Adjust to match basin_ids
                    if len(input_end_dates) < len(basin_ids):
                        input_end_dates = input_end_dates + [input_end_dates[-1]] * (
                            len(basin_ids) - len(input_end_dates)
                        )
                    else:
                        input_end_dates = input_end_dates[: len(basin_ids)]

                # Create expanded dates for each horizon - use current_horizons not self.horizons
                dates_expanded = []
                for i, input_date in enumerate(input_end_dates):
                    input_date_dt = pd.to_datetime(input_date)
                    for horizon in current_horizons:
                        # Calculate forecast date by adding horizon days to input end date
                        forecast_date = input_date_dt + pd.Timedelta(days=horizon)
                        dates_expanded.append(forecast_date)

                # Verify dates_expanded length matches other arrays
                assert len(dates_expanded) == len(preds_flat), (
                    f"dates_expanded length {len(dates_expanded)} doesn't match preds_flat length {len(preds_flat)}"
                )
            else:
                # Create dummy dates if not available
                print("Warning: No input_end_dates found, using dummy dates")
                dates_expanded = [pd.Timestamp.now()] * len(preds_flat)
        else:
            raise ValueError(
                f"Unexpected prediction shape {preds.shape}, expected 2D array [batch_size, pred_len]"
            )

        # Use the model-specific datamodule for inverse transforms if provided
        dm_for_transform = datamodule or self.default_datamodule
        if dm_for_transform and hasattr(
            dm_for_transform, "inverse_transform_predictions"
        ):
            try:
                preds_flat = dm_for_transform.inverse_transform_predictions(
                    preds_flat, basin_ids_expanded
                )
                obs_flat = dm_for_transform.inverse_transform_predictions(
                    obs_flat, basin_ids_expanded
                )
            except Exception as e:
                print(f"Warning: Failed to inverse transform predictions: {e}")

        # Create evaluation dataframe
        df = pd.DataFrame(
            {
                "horizon": horizons_expanded,
                "prediction": preds_flat,
                "observed": obs_flat,
                "basin_id": basin_ids_expanded,
                "date": dates_expanded,
            }
        )

        return df
