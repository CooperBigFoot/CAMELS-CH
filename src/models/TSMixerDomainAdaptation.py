import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl
from torch.nn import MSELoss
import math
from typing import Dict, Any, Union, List, Optional, Tuple
import numpy as np

from .TSMixer import TSMixer, TSMixerConfig


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class TSMixerDomainAdaptationConfig(TSMixerConfig):
    """Configuration for TSMixer with domain adaptation capabilities.

    Extends the base TSMixerConfig with domain adaptation specific parameters.
    """

    def __init__(
        self,
        input_len: int,
        input_size: int,
        output_len: int,
        static_size: int,
        hidden_size: int = 64,
        static_embedding_size: int = 10,
        num_layers: int = 5,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        group_identifier: str = "gauge_id",
        lr_scheduler_patience: int = 2,
        lr_scheduler_factor: float = 0.5,
        lambda_adv: float = 1.0,
        initial_lambda: float = 0.0,
        domain_loss_weight: float = 1.0,
        discriminator_hidden_dim: int = 16,

    ):
        """Initialize TSMixerDomainAdaptation configuration.

        Args:
            input_len: Length of input sequence (time steps)
            input_size: Number of input features per time step
            output_len: Length of output sequence to predict
            static_size: Number of static features
            hidden_size: Size of hidden layers in network
            static_embedding_size: Size of static feature embeddings
            num_layers: Number of ResBlock layers
            dropout: Dropout rate
            learning_rate: Initial learning rate
            group_identifier: Column name for the group/basin identifier
            lr_scheduler_patience: Patience for learning rate scheduler
            lr_scheduler_factor: Factor by which to reduce learning rate
            lambda_adv: Weight for gradient reversal scaling (final value)
            domain_loss_weight: Weight for domain loss in total loss function
            discriminator_hidden_dim: Hidden dimension size for domain discriminator

        """
        super().__init__(
            input_len=input_len,
            input_size=input_size,
            output_len=output_len,
            static_size=static_size,
            hidden_size=hidden_size,
            static_embedding_size=static_embedding_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            group_identifier=group_identifier,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor
        )

        self.lambda_adv = lambda_adv
        self.initial_lambda = initial_lambda
        self.domain_loss_weight = domain_loss_weight
        self.discriminator_hidden_dim = discriminator_hidden_dim


class LitTSMixerDomainAdaptation(pl.LightningModule):
    """PyTorch Lightning implementation of TSMixer with domain adaptation.

    This class extends the base TSMixer model with domain adaptation capabilities
    through adversarial training with a domain discriminator.
    """

    def __init__(
        self,
        config: Union[TSMixerDomainAdaptationConfig, Dict[str, Any]],
    ):
        """Initialize the domain adaptation model.

        Args:
            config: Either a TSMixerDomainAdaptationConfig object or a dictionary of config parameters
        """
        super().__init__()

        # Handle different config input types
        if isinstance(config, dict):
            # Create domain adaptation config if dictionary provided
            self.config = TSMixerDomainAdaptationConfig.from_dict(config)
        else:
            self.config = config

        # Create base TSMixer model
        self.model = TSMixer(self.config)

        # Domain adaptation components
        feature_dim = self.config.input_len * self.config.input_size
        self.domain_discriminator = DomainDiscriminator(
            feature_dim=feature_dim,
            hidden_dim=self.config.discriminator_hidden_dim
        )
        self.current_lambda = self.config.initial_lambda

        # Loss functions
        self.mse_criterion = MSELoss()
        self.domain_criterion = nn.BCELoss()

        # Save hyperparameters and initialize tracking variables
        self.save_hyperparameters(self.config.to_dict())
        self.test_outputs = []

    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        print("Backbone parameters frozen")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        print("Backbone parameters unfrozen")

    def freeze_head(self):
        """Freeze prediction head parameters."""
        for param in self.model.head.parameters():
            param.requires_grad = False
        print("Head parameters frozen")

    def unfreeze_head(self):
        """Unfreeze prediction head parameters."""
        for param in self.model.head.parameters():
            param.requires_grad = True
        print("Head parameters unfrozen")

    def extract_backbone_state_dict(self, state_dict):
        """Extract backbone-specific weights from a full model state dict."""
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.backbone.'):
                backbone_key = key.replace('model.backbone.', '')
                backbone_state_dict[backbone_key] = value
        return backbone_state_dict

    def load_backbone_from_pretrained(self, pretrained_state_dict):
        """Load only backbone weights from a pretrained model."""
        backbone_dict = self.extract_backbone_state_dict(pretrained_state_dict)
        self.model.backbone.load_state_dict(backbone_dict)
        print(f"Loaded backbone weights with {len(backbone_dict)} parameters")

    def load_from_pretrained(self, pretrained_state_dict):
        """Load the entire model weights from a pretrained TSMixer model."""
        model_state_dict = {}
        for key, value in pretrained_state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                model_state_dict[new_key] = value
        missing_keys, unexpected_keys = self.model.load_state_dict(
            model_state_dict, strict=False)
        if missing_keys:
            print(
                f"Warning: Missing keys when loading pretrained model: {missing_keys}")
        if unexpected_keys:
            print(
                f"Warning: Unexpected keys when loading pretrained model: {unexpected_keys}")
        print("Loaded model weights from pretrained model")

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Forward pass for time series prediction.

        Args:
            x: Dynamic input features [batch_size, seq_len, input_size]
            static: Static features [batch_size, static_size]

        Returns:
            Model predictions [batch_size, output_len, 1]
        """
        return self.model(x, static)

    def extract_features(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Extract domain-invariant features for adversarial training.

        The static features are explicitly zeroed out to ensure the domain classifier
        learns only from the dynamic features, preventing any static catchment 
        characteristics from influencing domain discrimination.

        Args:
            x: Dynamic input features [batch_size, seq_len, input_size]
            static: Static features [batch_size, static_size] (not used when zeroed)

        Returns:
            Flattened feature representations [batch_size, flattened_dim]
        """
        features = self.model.backbone(x, static, zero_static=True)
        flattened = features.flatten(start_dim=1)
        return flattened

    @staticmethod
    def get_lambda_value(progress: float) -> float:
        """
        Calculate lambda value based on training progress.

        Args:
            progress: Training progress from 0 to 1

        Returns:
            Current lambda value according to the paper's schedule

        Source:
            https://www.jmlr.org/papers/volume17/15-239/15-239.pdf
        """
        return 2 / (1 + np.exp(-10 * progress)) - 1

    def _get_scheduled_lambda(self) -> float:
        """Calculate lambda value based on current epoch and schedule type."""

        if not hasattr(self.trainer, "max_epochs") or self.trainer.max_epochs is None:
            print("Warning: trainer.max_epochs not set, using constant lambda")
            return self.config.lambda_adv

        current_epoch = self.trainer.current_epoch
        max_epochs = self.trainer.max_epochs
        progress = current_epoch / max_epochs

        return self.get_lambda_value(progress) * self.config.lambda_adv

    def on_train_epoch_start(self) -> None:
        """Update lambda value at the beginning of each epoch."""
        prev_lambda = self.current_lambda
        self.current_lambda = self._get_scheduled_lambda()

        self.log("lambda_adv", self.current_lambda, prog_bar=True)

    def training_step(self, batch, batch_idx):
        """Training step for domain adaptation.

        Processes both source and target domain data, computing task loss on source
        domain and domain adversarial loss on both domains.

        Args:
            batch: Combined batch containing source and target domain data
            batch_idx: Index of current batch

        Returns:
            Combined loss value
        """
        # Unpack the tuple from CombinedLoader
        data_dict, _, _ = batch  # (data_dict, 0, 0)
        source_batch = data_dict["source"]
        target_batch = data_dict["target"]

        # Combine data from both domains
        combined_X = torch.cat([source_batch["X"], target_batch["X"]])
        combined_static = torch.cat(
            [source_batch["static"], target_batch["static"]])
        combined_domain = torch.cat([
            torch.zeros(len(source_batch["X"])),  # Source domain = 0
            torch.ones(len(target_batch["X"]))    # Target domain = 1
        ]).unsqueeze(1).to(self.device)

        # Forward pass for task prediction (source domain only)
        y_pred = self(source_batch["X"], source_batch["static"])
        task_loss = self.mse_criterion(
            y_pred, source_batch["y"].unsqueeze(-1))

        # Forward pass for domain prediction (both domains)
        features = self.extract_features(combined_X, combined_static)
        features_rev = grad_reverse(features, self.current_lambda)
        domain_preds = self.domain_discriminator(features_rev)
        domain_loss = self.domain_criterion(domain_preds, combined_domain)

        # Total loss is a combination of task and domain losses
        total_loss = task_loss + self.config.domain_loss_weight * domain_loss

        self.log_dict({
            "train_loss": total_loss,
            "train_task_loss": task_loss,
            "train_domain_loss": domain_loss,
        }, batch_size=len(combined_X))  # Add explicit batch size

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step for domain adaptation.

        Evaluates both task performance on source domain and domain discrimination
        across both domains.

        Args:
            batch: Combined batch containing source and target domain data
            batch_idx: Index of current batch

        Returns:
            Dictionary with validation metrics
        """
        # Handle both CombinedLoader tuple and standard DataLoader dict formats
        data_dict, _, _ = batch  # CombinedLoader format
        source_batch = data_dict.get("source", {})
        target_batch = data_dict.get("target", {})

        metrics = {}

        # 1. Task Loss (source only)
        if "X" in source_batch and len(source_batch["X"]) > 0:
            source_pred = self(source_batch["X"], source_batch["static"])
            source_loss = self.mse_criterion(
                source_pred, source_batch["y"].unsqueeze(-1))
            metrics["val_loss"] = source_loss

        # 2. Domain Classification (only if both domains present)
        if "X" in source_batch and "X" in target_batch:
            combined_X = torch.cat([source_batch["X"], target_batch["X"]])
            combined_static = torch.cat(
                [source_batch["static"], target_batch["static"]])
            true_domains = torch.cat([
                torch.zeros(len(source_batch["X"])),
                torch.ones(len(target_batch["X"]))
            ]).unsqueeze(1).to(self.device)

            features = self.extract_features(combined_X, combined_static)
            domain_preds = self.domain_discriminator(features)
            domain_acc = ((domain_preds > 0.5).float()
                          == true_domains).float().mean()
            metrics["val_domain_acc"] = domain_acc

        self.log_dict(metrics, prog_bar=True, batch_size=len(combined_X))
        return metrics.get("val_loss", torch.tensor(0.0))

    def test_step(self, batch, batch_idx):
        """Test step for evaluating model performance.

        Args:
            batch: Batch of test data
            batch_idx: Index of current batch

        Returns:
            Dictionary with test outputs
        """
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch["static"]
        y_hat = self(x, static)

        output = {
            "predictions": y_hat.squeeze(-1),
            "observations": y.squeeze(-1),
            "basin_ids": batch[self.config.group_identifier],
        }

        self.test_outputs.append(output)
        return output

    def on_test_epoch_start(self) -> None:
        """Initialize storage for test results."""
        self.test_outputs = []

    def on_test_epoch_end(self) -> None:
        """Aggregate test results for evaluation."""
        if not self.test_outputs:
            print("Warning: No test outputs collected")
            return

        self.test_results = {
            "predictions": torch.cat([x["predictions"] for x in self.test_outputs]),
            "observations": torch.cat([x["observations"] for x in self.test_outputs]),
            "basin_ids": [bid for x in self.test_outputs for bid in x["basin_ids"]],
        }
        self.test_outputs = []

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.config.lr_scheduler_patience,
                factor=self.config.lr_scheduler_factor
            ),
            "monitor": "val_loss",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
