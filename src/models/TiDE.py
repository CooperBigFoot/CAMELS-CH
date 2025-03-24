"""
Based on the paper: "Long-term Forecasting with TiDE: Time-series Dense Encoder"
https://arxiv.org/pdf/2304.08424

Based code implementation from
https://github.com/thuml/Time-Series-Library/blob/main/models/TiDE.py
and
https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tide_model.html?highlight=tide#module-darts.models.forecasting.tide_model
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional


#########################################
# 1. Configuration class (TiDEConfig)
#########################################
class TiDEConfig:
    def __init__(
        self,
        input_len: int,
        output_len: int,
        output_size: int = 1,  # renamed from output_dim
        past_feature_size: int = 0,  # renamed from past_cov_dim
        future_input_size: int = 0,  # renamed from future_cov_dim
        static_size: int = 0,  # renamed from static_dim
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        decoder_output_size: int = 16,  # renamed from decoder_output_dim
        hidden_size: int = 128,
        temporal_decoder_hidden_size: int = 32,  # renamed from temporal_decoder_hidden
        past_feature_projection_size: int = 0,  # renamed from temporal_width_past
        future_forcing_projection_size: int = 0,  # renamed from temporal_width_future
        use_layer_norm: bool = False,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        # Legacy parameters (optional)
        output_dim: Optional[int] = None,
        past_cov_dim: Optional[int] = None,
        future_cov_dim: Optional[int] = None,
        static_dim: Optional[int] = None,
        decoder_output_dim: Optional[int] = None,
        temporal_decoder_hidden: Optional[int] = None,
        temporal_width_past: Optional[int] = None,
        temporal_width_future: Optional[int] = None,
    ):
        """
        Initialize TiDE configuration with standardized parameter naming.

        Args:
            input_len: Historical lookback window length.
            output_len: Forecast horizon.
            output_size: Number of target components.
            past_feature_size: Number of dynamic features besides target.
            future_input_size: Number of future forcing features.
            static_size: Dimension of static features.
            num_encoder_layers: Encoder layer count.
            num_decoder_layers: Decoder layer count.
            decoder_output_size: Decoder output size.
            hidden_size: Hidden layer size.
            temporal_decoder_hidden_size: Hidden size in temporal decoder.
            past_feature_projection_size: Projection size for past features.
            future_forcing_projection_size: Projection size for future forcing.
            use_layer_norm: Whether to apply LayerNorm.
            dropout: Dropout rate.
            learning_rate: Optimizer learning rate.
            (Legacy parameters are supported via deprecation warnings.)
        """
        self.input_len = input_len
        self.output_len = output_len
        self.output_size = output_size
        self.past_feature_size = past_feature_size
        self.future_input_size = future_input_size
        self.static_size = static_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_size = decoder_output_size
        self.hidden_size = hidden_size
        self.temporal_decoder_hidden_size = temporal_decoder_hidden_size
        self.past_feature_projection_size = past_feature_projection_size
        self.future_forcing_projection_size = future_forcing_projection_size
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        self.learning_rate = learning_rate

    def to_dict(self):
        return self.__dict__.copy()

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return self


#########################################
# 2. Residual Block for TiDE (TiDEResBlock)
#########################################
class TiDEResBlock(nn.Module):
    """
    A two-layer MLP with a skip connection and optional LayerNorm.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        dropout: float,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(input_dim, output_dim)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dense(x) + self.skip(x)
        if self.use_layer_norm:
            out = self.layer_norm(out)
        return out


#########################################
# 3. TiDE Model as an nn.Module (Backbone)
#########################################
class TiDEModel(nn.Module):
    """
    Implements a TiDE‐style encoder–decoder.

    Expects:
      • x: historical tensor of shape [B, input_len, target_dim + past_cov_dim].
      • future: future forcing features [B, output_len, future_cov_dim] (if any).
      • static: static features [B, static_dim] (if any).

    The encoder input is built by flattening and concatenating:
      - The target lookback (first output_dim channels).
      - Past covariates (if any; projected if temporal_width_past > 0).
      - Future forcing features (if any; projected if temporal_width_future > 0).
      - Static features (if any).
    """

    def __init__(self, config: TiDEConfig):
        super().__init__()
        self.config = config
        L = config.input_len
        H = config.output_len
        out_dim = config.output_size
        past_cov_dim = config.past_feature_size
        future_cov_dim = config.future_input_size
        static_dim = config.static_size

        # Corrected enc_dim calculation
        enc_dim = L * out_dim

        # Handle past covariates contribution
        if past_cov_dim > 0:
            if config.past_feature_projection_size > 0:
                past_contrib = L * config.past_feature_projection_size
            else:
                past_contrib = L * past_cov_dim
            enc_dim += past_contrib

        # Handle future covariates contribution
        if future_cov_dim > 0:
            if config.future_forcing_projection_size > 0:
                future_contrib = H * config.future_forcing_projection_size
            else:
                future_contrib = H * future_cov_dim
            enc_dim += future_contrib

        # Add static features
        if static_dim > 0:
            enc_dim += static_dim

        # Build the encoder: a stack of residual blocks.
        encoder_layers = []
        encoder_layers.append(
            TiDEResBlock(
                enc_dim,
                config.hidden_size,
                config.hidden_size,
                config.dropout,
                config.use_layer_norm,
            )
        )
        for _ in range(config.num_encoder_layers - 1):
            encoder_layers.append(
                TiDEResBlock(
                    config.hidden_size,
                    config.hidden_size,
                    config.hidden_size,
                    config.dropout,
                    config.use_layer_norm,
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)

        # Build the decoder: a stack of residual blocks.
        # Final layer outputs a vector of size = decoder_output_dim * H.
        decoder_layers = []
        for _ in range(config.num_decoder_layers - 1):
            decoder_layers.append(
                TiDEResBlock(
                    config.hidden_size,
                    config.hidden_size,
                    config.hidden_size,
                    config.dropout,
                    config.use_layer_norm,
                )
            )
        decoder_layers.append(
            TiDEResBlock(
                config.hidden_size,
                config.decoder_output_size * H,
                config.hidden_size,
                config.dropout,
                config.use_layer_norm,
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

        # Temporal decoder: fuses the decoder output with future features.
        # Its input dimension is decoder_output_dim plus (if future_cov_dim>0, projected future features).
        temporal_in_dim = config.decoder_output_size
        if future_cov_dim > 0:
            # If a projection is used, the future features are mapped to temporal_width_future.
            temporal_in_dim += (
                config.future_forcing_projection_size
                if config.future_forcing_projection_size > 0
                else future_cov_dim
            )
        self.temporal_decoder = TiDEResBlock(
            temporal_in_dim,
            out_dim,
            config.temporal_decoder_hidden_size,
            config.dropout,
            config.use_layer_norm,
        )

        # Lookback skip connection: projects the past target (x_target) from length L to H.
        self.lookback_skip = nn.Linear(L, H)

        # Optional projections for past and future covariates.
        if past_cov_dim > 0 and config.past_feature_projection_size > 0:
            self.past_projection = TiDEResBlock(
                past_cov_dim,
                config.past_feature_projection_size,
                config.hidden_size,
                config.dropout,
                config.use_layer_norm,
            )
        else:
            self.past_projection = None

        if future_cov_dim > 0 and config.future_forcing_projection_size > 0:
            self.future_projection = TiDEResBlock(
                future_cov_dim,
                config.future_forcing_projection_size,
                config.hidden_size,
                config.dropout,
                config.use_layer_norm,
            )
        else:
            self.future_projection = None

    def forward(
        self,
        x: torch.Tensor,
        future: torch.Tensor = None,
        static: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, input_len, target_dim + past_cov_dim].
            future: [B, output_len, future_cov_dim] (optional).
            static: [B, static_dim] (optional).
        Returns:
            Tensor of shape [B, output_len, output_dim].
        """
        B, L, _ = x.shape
        H = self.config.output_len
        out_dim = self.config.output_size

        # Split x into target and past covariates.
        x_target = x[:, :, :out_dim]  # [B, L, out_dim]
        if self.config.past_feature_size > 0:
            x_past = x[
                :, :, out_dim : out_dim + self.config.past_feature_size
            ]  # [B, L, past_cov_dim]
            if self.past_projection is not None:
                x_past = self.past_projection(x_past)  # [B, L, temporal_width_past]
        else:
            x_past = None

        # Process future covariates if provided.
        if self.config.future_input_size > 0 and future is not None:
            if self.future_projection is not None:
                future_proj = self.future_projection(
                    future
                )  # [B, H, temporal_width_future]
            else:
                future_proj = future  # [B, H, future_cov_dim]
        else:
            future_proj = None

        # Build encoder input by flattening and concatenating.
        enc_inputs = [x_target.reshape(B, -1)]
        if x_past is not None:
            enc_inputs.append(x_past.reshape(B, -1))
        if future_proj is not None:
            enc_inputs.append(future_proj.reshape(B, -1))
        if static is not None:
            enc_inputs.append(static)
        encoder_input = torch.cat(enc_inputs, dim=1)  # [B, enc_dim]

        # Pass through encoder and decoder.
        encoded = self.encoder(encoder_input)
        decoded = self.decoder(encoded)  # [B, decoder_output_dim * H]
        dec_out = decoded.reshape(B, H, -1)  # [B, H, decoder_output_dim]

        # Temporal decoding: fuse decoder output with (projected) future forcing if available.
        if future_proj is not None:
            temporal_input = torch.cat(
                [dec_out, future_proj], dim=-1
            )  # [B, H, decoder_output_dim + future_proj_dim]
        else:
            temporal_input = dec_out
        temporal_decoded = self.temporal_decoder(temporal_input)  # [B, H, out_dim]

        # Lookback skip: project the target history from length L to H.
        skip = self.lookback_skip(x_target.transpose(1, 2)).transpose(
            1, 2
        )  # [B, H, out_dim]

        # Final output: add skip connection.
        out = temporal_decoded + skip
        return out


#########################################
# 4. PyTorch Lightning Module (LitTiDE)
#########################################
class LitTiDE(pl.LightningModule):
    def __init__(self, config: TiDEConfig):
        super().__init__()
        # Save configuration as hyperparameters.
        self.save_hyperparameters(config.to_dict())
        self.model = TiDEModel(config)
        self.criterion = nn.MSELoss()
        self.learning_rate = config.learning_rate

        # Add tracking variables for test outputs
        self.test_outputs = []
        self.test_results = None

    def forward(self, x, future=None, static=None):
        return self.model(x, future, static)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute training step with TiDE model.

        Args:
            batch: Dictionary containing input data with keys 'X', 'y', and optionally 'future' and 'static'
            batch_idx: Index of the current batch

        Returns:
            Loss value for the batch
        """
        # Expect batch to be a dict with keys: 'X', 'future' (optional), 'static' (optional), 'y'
        x = batch["X"]
        future = batch.get("future", None)
        static = batch.get("static", None)
        y = batch["y"]
        y_hat = self(x, future, static)
        loss = self.criterion(y_hat, y.unsqueeze(-1))

        # Log metrics with batch size for proper averaging
        self.log("train_loss", loss, batch_size=x.size(0))
        self.log("train_mse", loss, batch_size=x.size(0))  # Same as loss in this case

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Execute validation step with TiDE model.

        Args:
            batch: Dictionary containing input data with keys 'X', 'y', and optionally 'future' and 'static'
            batch_idx: Index of the current batch

        Returns:
            Dictionary with validation metrics and predictions
        """
        x = batch["X"]
        future = batch.get("future", None)
        static = batch.get("static", None)
        y = batch["y"]
        y_hat = self(x, future, static)
        loss = self.criterion(y_hat, y.unsqueeze(-1))

        # Log metrics with batch size for proper averaging
        self.log("val_loss", loss, prog_bar=True, batch_size=x.size(0))
        self.log("val_mse", loss, batch_size=x.size(0))  # Same as loss in this case

        return {"val_loss": loss, "preds": y_hat, "targets": y}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Execute test step with TiDE model and collect outputs for evaluation.

        Args:
            batch: Dictionary containing input data with keys 'X', 'y', and optionally 'future' and 'static'
            batch_idx: Index of the current batch

        Returns:
            Dictionary with test outputs for evaluation
        """
        x = batch["X"]
        future = batch.get("future", None)
        static = batch.get("static", None)
        y = batch["y"]
        y_hat = self(x, future, static)
        loss = self.criterion(y_hat, y.unsqueeze(-1))

        # Log test metrics
        self.log("test_loss", loss, batch_size=x.size(0))
        self.log("test_mse", loss, batch_size=x.size(0))

        # Create output dictionary for evaluation
        output = {
            "predictions": y_hat.squeeze(-1),  # [batch_size, output_len]
            "observations": y.squeeze(-1),  # [batch_size, output_len]
            "basin_ids": batch.get(
                "gauge_id", batch.get("basin_id", None)
            ),  # Support both naming conventions
        }

        # Add optional metadata if available in the batch
        if "input_end_date" in batch:
            output["input_end_date"] = batch["input_end_date"]
        if "slice_idx" in batch:
            output["slice_idx"] = batch["slice_idx"]

        # Collect outputs for evaluation
        self.test_outputs.append(output)

        return output

    def on_test_epoch_start(self) -> None:
        """Reset test outputs collector at the beginning of test epoch."""
        self.test_outputs = []

    def on_test_epoch_end(self) -> None:
        """Consolidate all test outputs at the end of test epoch."""
        if not self.test_outputs:
            print("Warning: No test outputs collected")
            return

        # Consolidate all test outputs into a single dictionary
        self.test_results = {
            "predictions": torch.cat([x["predictions"] for x in self.test_outputs]),
            "observations": torch.cat([x["observations"] for x in self.test_outputs]),
            "basin_ids": [bid for x in self.test_outputs for bid in x["basin_ids"]],
        }

        # Add optional metadata if available
        if "input_end_date" in self.test_outputs[0]:
            self.test_results["input_end_date"] = [
                date for x in self.test_outputs for date in x["input_end_date"]
            ]

        if "slice_idx" in self.test_outputs[0]:
            self.test_results["slice_idx"] = [
                idx for x in self.test_outputs for idx in x["slice_idx"]
            ]

        # Clear temporary storage
        self.test_outputs = []

    def configure_optimizers(self):
        """Configure the optimizer for model training.

        Returns:
            PyTorch optimizer instance
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
