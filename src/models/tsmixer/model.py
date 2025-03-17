"""
TSMixer model implementation based on the paper:
"TSMixer: An All-MLP Architecture for Time Series Forecasting"
https://arxiv.org/abs/2303.06053
"""

from typing import Optional
import torch
import torch.nn as nn
from .config import TSMixerConfig


class InputAlignmentModule(nn.Module):
    """Aligns historical data, future forcing, and static features into a common representation.

    This module implements the "align" stage described in Section 4.2 of the TSMixer paper,
    projecting heterogeneous inputs into a unified space for subsequent mixing layers.
    """

    def __init__(
        self,
        input_size: int,
        input_len: int,
        output_len: int,
        future_input_size: int,
        hidden_size: int,
        static_size: int,
        static_embedding_size: int,
        dropout: float = 0.1,
        fusion_method: str = "add",
    ):
        """Initialize the alignment module.

        Args:
            input_size: Number of input features
            input_len: Length of input sequence
            output_len: Length of output sequence (forecast horizon)
            future_input_size: Number of future forcing features
            hidden_size: Size of hidden representation
            static_size: Number of static features
            static_embedding_size: Size of static feature embedding
            dropout: Dropout rate for regularization
            fusion_method: Method to fuse representations ("add" or "concat")
        """
        super().__init__()

        self.fusion_method = fusion_method
        self.output_len = output_len

        # Historical data projection
        self.historical_projection = nn.Sequential(
            nn.Linear(input_len * input_size, hidden_size * output_len),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Future forcing projection
        self.future_projection = nn.Sequential(
            nn.Linear(future_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Static feature projection (if available)
        if static_size > 0:
            self.static_projection = nn.Sequential(
                nn.Linear(static_size, static_embedding_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Gate for static modulation
            self.static_gate = nn.Linear(static_embedding_size, hidden_size)
        else:
            self.static_projection = None
            self.static_gate = None

        # Final output size after fusion
        if fusion_method == "add":
            self.output_size = hidden_size
        elif fusion_method == "concat":
            self.output_size = hidden_size * 2
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

        # Layer normalization for aligned representations
        self.norm_historical = nn.LayerNorm(hidden_size)
        self.norm_future = nn.LayerNorm(hidden_size)
        self.norm_fused = nn.LayerNorm(self.output_size)

    def forward(
        self,
        historical: torch.Tensor,
        future: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Align and fuse historical, future, and static data.

        Args:
            historical: Historical data [batch_size, input_len, input_size]
            future: Future forcing data [batch_size, output_len, future_input_size]
            static: Static features [batch_size, static_size]

        Returns:
            Fused representation [batch_size, output_len, output_size]
        """
        batch_size = historical.size(0)

        # Project historical data to match output sequence length
        hist_flat = historical.reshape(batch_size, -1)  # [B, input_len * input_size]
        hist_proj = self.historical_projection(
            hist_flat
        )  # [B, hidden_size * output_len]
        hist_aligned = hist_proj.reshape(
            batch_size, self.output_len, -1
        )  # [B, output_len, hidden_size]
        hist_aligned = self.norm_historical(hist_aligned)

        # Project future forcing data
        future_aligned = self.norm_future(
            self.future_projection(future)
        )  # [B, output_len, hidden_size]

        # Apply static modulation if available
        if static is not None and self.static_projection is not None:
            static_emb = self.static_projection(static)  # [B, static_embedding_size]
            static_gate = torch.sigmoid(
                self.static_gate(static_emb)
            )  # [B, hidden_size]
            static_gate = static_gate.unsqueeze(1).expand(
                -1, self.output_len, -1
            )  # [B, output_len, hidden_size]

            # Apply modulation
            hist_aligned = hist_aligned * static_gate
            future_aligned = future_aligned * static_gate

        # Fuse representations
        if self.fusion_method == "add":
            fused = hist_aligned + future_aligned
        else:  # concat
            fused = torch.cat([hist_aligned, future_aligned], dim=-1)

        return self.norm_fused(fused)


class FeatureMixingBlock(nn.Module):
    """Feature mixing block that processes each time step across features."""

    def __init__(self, input_dim: int, hidden_size: int, dropout: float):
        """Initialize feature mixing block.

        Args:
            input_dim: Dimension of input features
            hidden_size: Size of hidden representation
            dropout: Dropout rate
        """
        super().__init__()

        self.mixing = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature mixing block.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Mixed tensor [batch_size, seq_len, input_dim]
        """
        return self.norm(x + self.mixing(x))


class TimeMixingBlock(nn.Module):
    """Time mixing block that processes each feature across time."""

    def __init__(self, input_len: int, hidden_size: int, dropout: float):
        """Initialize time mixing block.

        Args:
            input_len: Length of input sequence
            hidden_size: Size of hidden representation
            dropout: Dropout rate
        """
        super().__init__()

        self.mixing = nn.Sequential(
            nn.Linear(input_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_len),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(input_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through time mixing block.

        Args:
            x: Input tensor [batch_size, seq_len, feature_dim]

        Returns:
            Mixed tensor [batch_size, seq_len, feature_dim]
        """
        # Transpose to apply mixing along time dimension
        x_t = x.transpose(1, 2)
        mixed = x_t + self.mixing(x_t)
        # Transpose back to original shape
        return self.norm(mixed).transpose(1, 2)


class ResBlock(nn.Module):
    """Residual block combining temporal and feature mixing."""

    def __init__(
        self, input_dim: int, hidden_size: int, dropout: float, input_len: int
    ):
        """Initialize residual block.

        Args:
            input_dim: Dimension of input features
            hidden_size: Size of hidden layers
            dropout: Dropout rate
            input_len: Length of input sequence
        """
        super().__init__()

        # Temporal mixing: mixing along the time dimension
        self.temporal = TimeMixingBlock(input_len, hidden_size, dropout)

        # Channel (feature) mixing: mixing along the feature dimension
        self.channel = FeatureMixingBlock(input_dim, hidden_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Output tensor [batch_size, seq_len, input_dim]
        """
        # Temporal mixing
        x = self.temporal(x)

        # Channel mixing
        x = self.channel(x)

        return x


class ConditionalFeatureMixing(nn.Module):
    """Applies conditional feature mixing using static features to modulate dynamic features."""

    def __init__(
        self,
        input_size: int,
        static_size: int,
        static_embedding_size: int,
        hidden_size: int,
    ):
        """Initialize conditional feature mixing module.

        Args:
            input_size: Dimension of input features
            static_size: Dimension of static features
            static_embedding_size: Size of static feature embedding
            hidden_size: Size of hidden layers
        """
        super().__init__()

        # Projections for static features
        self.static_proj = nn.Linear(static_size, static_embedding_size)

        # Gate projection for modulation
        self.gate_proj = nn.Linear(static_embedding_size, input_size)

        # Feature mixing after modulation
        self.feature_mixing = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

        self.norm = nn.LayerNorm(input_size)

    def forward(self, x_dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Forward pass for conditional feature mixing.

        Args:
            x_dynamic: Dynamic input features [batch_size, seq_len, input_size]
            static: Static features [batch_size, static_size]

        Returns:
            Modulated features [batch_size, seq_len, input_size]
        """
        # Project static features
        static_emb = self.static_proj(static)

        # Expand static features across time dimension
        static_expanded = static_emb.unsqueeze(1).expand(-1, x_dynamic.size(1), -1)

        # Create modulation gate
        gate = torch.sigmoid(self.gate_proj(static_expanded))

        # Apply modulation to dynamic features
        x_conditioned = x_dynamic * gate

        # Apply feature mixing
        mixed = self.feature_mixing(x_conditioned)

        # Apply residual connection and normalization
        return self.norm(x_conditioned + mixed)


class TSMixerBackbone(nn.Module):
    """Enhanced TSMixerBackbone with input alignment and future forcing integration."""

    def __init__(self, config: TSMixerConfig):
        """Initialize TSMixer backbone.

        Args:
            config: TSMixer configuration
        """
        super().__init__()

        # Input alignment module to integrate historical, future, and static features
        self.alignment_module = InputAlignmentModule(
            input_size=config.input_size,
            input_len=config.input_len,
            output_len=config.output_len,
            future_input_size=config.future_input_size,
            hidden_size=config.hidden_size,
            static_size=config.static_size,
            static_embedding_size=config.static_embedding_size,
            dropout=config.dropout,
            fusion_method=config.fusion_method,
        )

        # Determine the input dimension for mixing layers based on fusion method
        input_dim = (
            config.hidden_size * 2
            if config.fusion_method == "concat"
            else config.hidden_size
        )

        # Main mixing layers
        self.layers = nn.ModuleList(
            [
                ResBlock(
                    input_dim=input_dim,
                    hidden_size=config.hidden_size,
                    dropout=config.dropout,
                    input_len=config.output_len,
                )
                for _ in range(config.num_mixing_layers)
            ]
        )

        # Optional conditional feature mixing (kept for backward compatibility)
        self.conditional_mixing = ConditionalFeatureMixing(
            input_size=config.input_size,
            static_size=config.static_size,
            static_embedding_size=config.static_embedding_size,
            hidden_size=config.hidden_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor,
        future: Optional[torch.Tensor] = None,
        zero_static: bool = False,
        use_legacy: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with input alignment and future forcing integration.

        Args:
            x: Dynamic input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Future forcing data [batch_size, output_len, future_input_size]
            zero_static: If True, bypass static feature conditioning
            use_legacy: If True, use the legacy implementation without future data

        Returns:
            Processed tensor [batch_size, output_len, feature_dim]
        """
        if use_legacy or future is None:
            # Legacy mode (backward compatibility)
            if not zero_static:
                # Apply conditional feature mixing
                x = self.conditional_mixing(x, static)

            # Process through mixing layers
            for layer in self.layers:
                x = layer(x)

            return x

        # Modern mode with future forcing
        # Align and fuse historical data with future forcing
        fused = self.alignment_module(
            historical=x, future=future, static=None if zero_static else static
        )

        # Process through mixing layers
        for layer in self.layers:
            fused = layer(fused)

        return fused


class TSMixerHead(nn.Module):
    """Prediction head for TSMixer."""

    def __init__(
        self, input_dim: int, input_len: int, hidden_size: int, output_len: int
    ):
        """Initialize TSMixer head.

        Args:
            input_dim: Dimension of input features
            input_len: Length of input sequence
            hidden_size: Size of hidden layers
            output_len: Length of output sequence
        """
        super().__init__()

        # Adjust the head to work with aligned data
        self.output_len = output_len
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features to predictions.

        Args:
            x: Input features [batch_size, seq_len, input_dim]

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        # Apply projection to each time step independently
        return self.projection(x)  # [B, output_len, 1]


class TSMixer(nn.Module):
    """Complete TSMixer model with future forcing integration."""

    def __init__(self, config: TSMixerConfig):
        """Initialize TSMixer model.

        Args:
            config: TSMixer configuration
        """
        super().__init__()

        self.backbone = TSMixerBackbone(config)

        # Determine input dimension for head based on fusion method
        head_input_dim = (
            config.hidden_size * 2
            if config.fusion_method == "concat"
            else config.hidden_size
        )

        self.head = TSMixerHead(
            input_dim=head_input_dim,
            input_len=config.output_len,
            hidden_size=config.hidden_size,
            output_len=config.output_len,
        )
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with support for future forcing data.

        Args:
            x: Historical input features [B, input_len, input_size] 
               (contains target as the first feature, followed by optional past features)
            static: Static features [B, static_size]
            future: Optional future forcing data [B, output_len, future_input_size]

        Returns:
            Predictions [B, output_len, 1]
        """
        # Validate input dimensions
        assert x.ndim == 3, "Input tensor x must be of shape [B, input_len, input_size]"
        if static is not None:
            assert static.ndim == 2, "Static tensor must be of shape [B, static_size]"
        if future is not None:
            assert future.ndim == 3, (
                "Future tensor must be of shape [B, output_len, future_input_size]"
            )
        features = self.backbone(x, static, future)
        return self.head(features)
