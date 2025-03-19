import torch
import torch.nn as nn
from typing import Optional
from .config import TFTConfig


class GLU(nn.Module):
    """Gated Linear Unit for controlling information flow.

    As described in the paper, GLU allows the network to control the extent
    to which different components contribute to the output.
    """

    def __init__(self, input_size: int):
        """Initialize the GLU.

        Args:
            input_size: Dimension of input features
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gating mechanism.

        Args:
            x: Input tensor [batch_size, ..., input_size]

        Returns:
            Gated output [batch_size, ..., input_size]
        """
        sig = self.sigmoid(self.fc1(x))
        x_proj = self.fc2(x)
        return torch.mul(sig, x_proj)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network as described in the TFT paper.

    The GRN allows the model to skip over unused components and create
    variable-depth networks. It includes dropout for regularization
    and layer normalization.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        context_size: Optional[int] = None,
        residual: bool = True,
    ):
        """Initialize GRN.

        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden layer
            output_size: Dimension of output features
            dropout: Dropout rate
            context_size: Dimension of context vector (if provided)
            residual: Whether to use residual connection
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        self.residual = residual

        # Input projection when dimensions don't match for the residual
        if self.input_size != self.output_size and residual:
            self.skip_layer = nn.Linear(self.input_size, self.output_size)

        # Main network layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        # Context projection if provided
        if self.context_size is not None:
            self.context_layer = nn.Linear(
                self.context_size, self.hidden_size, bias=False
            )

        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)

        # Gating and normalization
        self.gate = GLU(self.output_size)
        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the GRN.

        Args:
            x: Input tensor [batch_size, ..., input_size]
            context: Optional context tensor [batch_size, ..., context_size]

        Returns:
            Output tensor [batch_size, ..., output_size]
        """
        # Prepare skip connection if dimensions differ
        if self.input_size != self.output_size and self.residual:
            residual = self.skip_layer(x)
        else:
            residual = x if self.residual else 0

        # First dense layer
        hidden = self.fc1(x)

        # Add context if provided
        if self.context_size is not None and context is not None:
            hidden = hidden + self.context_layer(context)

        # Nonlinearity
        hidden = self.elu(hidden)

        # Second dense layer with dropout
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        # Gating
        gated_hidden = self.gate(hidden)

        # Skip connection and normalization
        out = self.layer_norm(gated_hidden + residual)

        return out


class VariableSelectionNetwork(nn.Module):
    """Variable selection network as introduced in the TFT paper.

    This network provides instance-wise variable selection, determining
    which variables are most relevant for the prediction task.
    """

    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_size: int,
        dropout: float,
        context_size: Optional[int] = None,
    ):
        """Initialize variable selection network.

        Args:
            input_dim: Dimension of each input variable
            num_inputs: Number of input variables
            hidden_size: Dimension of hidden layers
            dropout: Dropout rate
            context_size: Dimension of context vector (optional)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.num_inputs = num_inputs

        # Network for variable selection weights
        if context_size is not None:
            self.selection_network = GatedResidualNetwork(
                input_size=self.num_inputs * self.input_dim,
                hidden_size=self.hidden_size,
                output_size=self.num_inputs,
                dropout=dropout,
                context_size=context_size,
            )
        else:
            self.selection_network = GatedResidualNetwork(
                input_size=self.num_inputs * self.input_dim,
                hidden_size=self.hidden_size,
                output_size=self.num_inputs,
                dropout=dropout,
            )

        # Separate GRNs for each input variable
        self.var_processors = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_size=self.input_dim,
                    hidden_size=self.hidden_size,
                    output_size=self.hidden_size,
                    dropout=dropout,
                )
                for _ in range(self.num_inputs)
            ]
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, flattened_embedding: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through variable selection network.

        Args:
            flattened_embedding: Flattened variable embeddings
                                [batch_size, ..., num_inputs * input_dim]
            context: Optional context tensor [batch_size, ..., context_size]

        Returns:
            Tuple containing:
                processed_embeddings: Processed variable embeddings
                                    [batch_size, ..., hidden_size]
                sparse_weights: Variable selection weights
                               [batch_size, ..., num_inputs]
        """
        # Calculate variable selection weights
        if context is not None:
            sparse_weights = self.selection_network(flattened_embedding, context)
        else:
            sparse_weights = self.selection_network(flattened_embedding)

        sparse_weights = self.softmax(sparse_weights)

        # Process each variable with its GRN
        var_outputs = []
        for i in range(self.num_inputs):
            # Extract embedding for specific variable
            var_embedding = flattened_embedding[
                ..., (i * self.input_dim) : ((i + 1) * self.input_dim)
            ]
            var_outputs.append(self.var_processors[i](var_embedding))

        var_outputs = torch.stack(var_outputs, dim=-1)

        # Weight each variable's processed representation by selection weights
        combined_outputs = torch.sum(var_outputs * sparse_weights.unsqueeze(-2), dim=-1)

        return combined_outputs, sparse_weights


class StaticCovariateEncoder(nn.Module):
    """Static covariate encoder as described in the TFT paper.

    This encoder processes static metadata and generates context vectors
    for various parts of the network.
    """

    def __init__(self, hidden_size: int, static_input_size: int, dropout: float):
        """Initialize static covariate encoder.

        Args:
            hidden_size: Dimension of hidden layers
            static_input_size: Dimension of static inputs after initial processing
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_size = hidden_size

        # Create GRNs for each context vector
        # c_s: Context for variable selection
        self.grn_vs = GatedResidualNetwork(
            input_size=static_input_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=dropout,
        )

        # c_e: Context for static enrichment
        self.grn_se = GatedResidualNetwork(
            input_size=static_input_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=dropout,
        )

        # c_h, c_c: Context vectors for LSTM initialization
        self.grn_h = GatedResidualNetwork(
            input_size=static_input_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=dropout,
        )

        self.grn_c = GatedResidualNetwork(
            input_size=static_input_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=dropout,
        )

    def forward(self, static_embedding: torch.Tensor) -> dict:
        """Process static covariates and return context vectors.

        Args:
            static_embedding: Static embeddings [batch_size, static_input_size]

        Returns:
            Dictionary containing context vectors:
                c_s: Context for variable selection
                c_e: Context for static enrichment
                c_h: Context for LSTM hidden state
                c_c: Context for LSTM cell state
        """
        contexts = {
            "c_s": self.grn_vs(static_embedding),  # Variable selection context
            "c_e": self.grn_se(static_embedding),  # Static enrichment context
            "c_h": self.grn_h(static_embedding),  # LSTM hidden state context
            "c_c": self.grn_c(static_embedding),  # LSTM cell state context
        }

        return contexts


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention as described in the TFT paper.

    This is a modified version of the standard multi-head attention that
    enables interpretability by sharing values across heads.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        """Initialize interpretable multi-head attention.

        Args:
            hidden_size: Dimension of input and output features
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        # Dimension per head
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            "hidden_size must be divisible by num_heads"
        )

        # Linear layers for queries and keys (head-specific)
        self.q_projections = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.head_dim, bias=False)
                for _ in range(self.num_heads)
            ]
        )

        self.k_projections = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.head_dim, bias=False)
                for _ in range(self.num_heads)
            ]
        )

        # Shared value projection across all heads
        self.v_projection = nn.Linear(hidden_size, self.head_dim, bias=False)

        # Output projection
        self.output_projection = nn.Linear(self.head_dim, hidden_size)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Forward pass of interpretable multi-head attention.

        Args:
            q: Query tensor [batch_size, target_len, hidden_size]
            k: Key tensor [batch_size, source_len, hidden_size]
            v: Value tensor [batch_size, source_len, hidden_size]
            mask: Optional attention mask [batch_size, target_len, source_len]

        Returns:
            Tuple containing:
                output: Attention output [batch_size, target_len, hidden_size]
                attention_weights: Attention weights [batch_size, target_len, source_len]
        """
        batch_size, target_len, _ = q.size()
        source_len = k.size(1)

        # Project values once (shared across heads)
        v_proj = self.v_projection(v)  # [batch_size, source_len, head_dim]

        # Initialize attention weights accumulator
        attn_weights_per_head = []

        # Process each head
        for head_idx in range(self.num_heads):
            # Project queries and keys for this head
            q_proj = self.q_projections[head_idx](
                q
            )  # [batch_size, target_len, head_dim]
            k_proj = self.k_projections[head_idx](
                k
            )  # [batch_size, source_len, head_dim]

            # Compute attention scores
            attn_scores = torch.bmm(
                q_proj,  # [batch_size, target_len, head_dim]
                k_proj.transpose(1, 2),  # [batch_size, head_dim, source_len]
            )  # [batch_size, target_len, source_len]

            # Scale attention scores
            attn_scores = attn_scores / (self.head_dim**0.5)

            # Apply mask if provided
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

            # Apply softmax to get attention weights
            attn_weights = torch.softmax(
                attn_scores, dim=-1
            )  # [batch_size, target_len, source_len]
            attn_weights = self.attn_dropout(attn_weights)

            attn_weights_per_head.append(attn_weights)

        # Average attention weights across heads
        avg_attn_weights = torch.stack(attn_weights_per_head).mean(
            dim=0
        )  # [batch_size, target_len, source_len]

        # Apply attention weights to values
        context = torch.bmm(
            avg_attn_weights,  # [batch_size, target_len, source_len]
            v_proj,  # [batch_size, source_len, head_dim]
        )  # [batch_size, target_len, head_dim]

        # Apply output projection
        output = self.output_projection(
            context
        )  # [batch_size, target_len, hidden_size]

        return output, avg_attn_weights


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer model for multi-horizon forecasting.

    The TFT model integrates multi-horizon forecasting with interpretable
    insights into temporal dynamics using variable selection, static covariate
    encoding, and interpretable multi-head attention.
    """

    def __init__(self, config: TFTConfig):
        """Initialize the TFT model.

        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.config = config

        # Dimensions and parameters
        self.input_size = config.input_size
        self.static_size = config.static_size
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.future_input_size = config.future_input_size or max(
            1, config.input_size - 1
        )

        # Input processing
        # We use linear layers for continuous variables
        self.target_input_proj = nn.Linear(1, self.hidden_size)

        if self.input_size > 1:  # If we have other features besides the target
            self.past_feat_proj = nn.Linear(self.input_size - 1, self.hidden_size)

        if self.future_input_size > 0:
            self.future_feat_proj = nn.Linear(self.future_input_size, self.hidden_size)

        if self.static_size > 0:
            self.static_feat_proj = nn.Linear(self.static_size, self.hidden_size)

        # Variable selection networks
        self.past_vsn = VariableSelectionNetwork(
            input_dim=self.hidden_size,
            num_inputs=2,  # Target and past features
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            context_size=self.hidden_size if self.static_size > 0 else None,
        )

        # Variable selection network for future (known) inputs
        if self.future_input_size > 0:
            self.future_vsn = VariableSelectionNetwork(
                input_dim=self.hidden_size,
                num_inputs=1,  # Future features
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                context_size=self.hidden_size if self.static_size > 0 else None,
            )

        # Static covariate encoder (if static inputs are provided)
        if self.static_size > 0:
            self.static_encoder = StaticCovariateEncoder(
                hidden_size=self.hidden_size,
                static_input_size=self.hidden_size,
                dropout=self.dropout,
            )

        # LSTM encoder-decoder for local processing
        self.lstm_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=config.lstm_layers,
            dropout=self.dropout if config.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=config.lstm_layers,
            dropout=self.dropout if config.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # Post-LSTM gating
        self.post_lstm_gate = GLU(self.hidden_size)
        self.post_lstm_norm = nn.LayerNorm(self.hidden_size)

        # Static enrichment
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            context_size=self.hidden_size if self.static_size > 0 else None,
        )

        # Temporal self-attention layer
        self.attention = InterpretableMultiHeadAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attn_dropout,
        )

        # Post-attention gating
        self.post_attn_gate = GLU(self.hidden_size)
        self.post_attn_norm = nn.LayerNorm(self.hidden_size)

        # Position-wise feed-forward network
        self.pos_wise_ff = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # Final gating
        self.pre_output_gate = GLU(self.hidden_size)
        self.pre_output_norm = nn.LayerNorm(self.hidden_size)

        # Output layer
        self.output_layer = nn.Linear(self.hidden_size, 1)  # Single quantile for now

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the TFT model.

        Args:
            x: Historical input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Future forcing data [batch_size, output_len, future_input_size]

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        batch_size = x.size(0)

        # Handle future inputs if not provided
        if future is None and self.future_input_size > 0:
            future = torch.zeros(
                batch_size, self.output_len, self.future_input_size, device=x.device
            )

        # Split x into target and features if needed
        # Assuming the first feature is the target
        target = x[:, :, 0:1]  # [batch_size, input_len, 1]

        # Process inputs
        target_emb = self.target_input_proj(
            target
        )  # [batch_size, input_len, hidden_size]

        if self.input_size > 1:
            past_feats = x[:, :, 1:]  # [batch_size, input_len, input_size-1]
            past_feats_emb = self.past_feat_proj(
                past_feats
            )  # [batch_size, input_len, hidden_size]
        else:
            # Create a dummy past features embedding if we only have the target
            past_feats_emb = torch.zeros_like(target_emb)

        if self.future_input_size > 0 and future is not None:
            future_feats_emb = self.future_feat_proj(
                future
            )  # [batch_size, output_len, hidden_size]

        # Process static inputs if provided
        static_context = None
        if self.static_size > 0 and static is not None:
            static_emb = self.static_feat_proj(static)  # [batch_size, hidden_size]
            static_context = self.static_encoder(static_emb)

        # Variable selection for past inputs
        # Combine target embedding and past features embedding
        past_inputs = torch.cat(
            [
                target_emb.reshape(batch_size, self.input_len, 1, self.hidden_size),
                past_feats_emb.reshape(batch_size, self.input_len, 1, self.hidden_size),
            ],
            dim=2,
        )  # [batch_size, input_len, 2, hidden_size]

        # Flatten the variable dimension for the variable selection network
        past_flattened = past_inputs.reshape(
            batch_size, self.input_len, -1
        )  # [batch_size, input_len, 2*hidden_size]

        # Apply variable selection network
        past_transformed = []
        for t in range(self.input_len):
            # For each timestep, apply the VSN
            c_s = static_context["c_s"] if static_context is not None else None
            timestep_embedding, _ = self.past_vsn(past_flattened[:, t, :], c_s)
            past_transformed.append(timestep_embedding)

        past_transformed = torch.stack(
            past_transformed, dim=1
        )  # [batch_size, input_len, hidden_size]

        # Variable selection for future inputs (if available)
        if self.future_input_size > 0 and future is not None:
            future_inputs = future_feats_emb.unsqueeze(
                2
            )  # [batch_size, output_len, 1, hidden_size]
            future_flattened = future_inputs.reshape(
                batch_size, self.output_len, -1
            )  # [batch_size, output_len, hidden_size]

            future_transformed = []
            for t in range(self.output_len):
                # For each timestep, apply the VSN
                c_s = static_context["c_s"] if static_context is not None else None
                timestep_embedding, _ = self.future_vsn(future_flattened[:, t, :], c_s)
                future_transformed.append(timestep_embedding)

            future_transformed = torch.stack(
                future_transformed, dim=1
            )  # [batch_size, output_len, hidden_size]
        else:
            future_transformed = torch.zeros(
                batch_size, self.output_len, self.hidden_size, device=x.device
            )

        # LSTM encoding (historical)
        init_hidden = None
        if static_context is not None:
            # Initialize LSTM states with static context if available
            h0 = (
                static_context["c_h"].unsqueeze(0).repeat(self.config.lstm_layers, 1, 1)
            )
            c0 = (
                static_context["c_c"].unsqueeze(0).repeat(self.config.lstm_layers, 1, 1)
            )
            init_hidden = (h0, c0)

        historical_features, lstm_state = self.lstm_encoder(
            past_transformed, init_hidden
        )

        # LSTM decoding (future)
        future_features, _ = self.lstm_decoder(future_transformed, lstm_state)

        # Temporal features from LSTM
        temporal_features = torch.cat([historical_features, future_features], dim=1)

        # Skip connection over the LSTM
        temporal_inputs = torch.cat([past_transformed, future_transformed], dim=1)
        temporal_features = self.post_lstm_gate(temporal_features)
        temporal_features = self.post_lstm_norm(temporal_features + temporal_inputs)

        # Static enrichment
        c_e = static_context["c_e"] if static_context is not None else None
        enriched_timesteps = []
        for i in range(temporal_features.shape[1]):
            # Process each timestep individually
            timestep = temporal_features[:, i, :]
            enriched_timestep = self.static_enrichment(timestep, c_e)
            enriched_timesteps.append(enriched_timestep)

        # Stack the processed timesteps to create the enriched tensor
        enriched = torch.stack(enriched_timesteps, dim=1)

        # Temporal self-attention (decoder masking)
        # Create causal mask to prevent attending to future timesteps
        seq_len = self.input_len + self.output_len
        mask = torch.ones(seq_len, seq_len, device=x.device).triu_(diagonal=1).bool()
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        attn_output, _ = self.attention(enriched, enriched, enriched, mask=mask)

        # Extract the future part for prediction (we only care about the output horizon)
        attn_output = attn_output[:, self.input_len :, :]
        enriched_future = enriched[:, self.input_len :, :]

        # Skip connection over attention
        attn_output = self.post_attn_gate(attn_output)
        attn_output = self.post_attn_norm(attn_output + enriched_future)

        # Position-wise feed-forward
        output = self.pos_wise_ff(attn_output)

        # Final skip connection
        decoder_output = temporal_features[:, self.input_len :, :]
        output = self.pre_output_gate(output)
        output = self.pre_output_norm(output + decoder_output)

        # Final linear projection to output
        output = self.output_layer(output)  # [batch_size, output_len, 1]

        return output
