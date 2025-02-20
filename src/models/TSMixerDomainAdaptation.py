import torch.nn as nn
import torch
from torch.optim import Adam
from .TSMixer import TSMixer
import pytorch_lightning as pl
from torch.nn import MSELoss


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
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class LitTSMixerDomainAdaptation(pl.LightningModule):
    def __init__(self, config, lambda_adv=1.0, domain_loss_weight=1.0, learning_rate=1e-3, group_identifier="basin_id"):
        super().__init__()
        self.save_hyperparameters()

        # Original TSMixer components
        self.config = config
        self.model = TSMixer(config)
        self.mse_criterion = MSELoss()

        # Domain adaptation components
        feature_dim = config.input_len * \
            (config.input_size + config.static_embedding_size)
        self.domain_discriminator = DomainDiscriminator(feature_dim)
        self.domain_criterion = nn.BCELoss()

        self.learning_rate = learning_rate
        self.group_identifier = group_identifier

    def forward(self, x, static):
        return self.model(x, static)

    def extract_features(self, x, static):

        # TODO: Potential domain leackage through static features. If model performs poorly
        # on target domain, consider removing static features from adversarial training.
        features = self.model.backbone(x, static)
        return features.flatten(start_dim=1)

    def training_step(self, batch, batch_idx):
        x, y, static = batch["X"], batch["y"].unsqueeze(-1), batch["static"]
        domain_labels = batch["domain_id"]

        # Ensure domain_labels is a tensor and has correct shape
        if not isinstance(domain_labels, torch.Tensor):
            domain_labels = torch.tensor(domain_labels, device=self.device)

        # Squeeze out extra dimensions and convert to float
        domain_labels = domain_labels.squeeze().float()

        # Create source mask - source domain has label 0
        # Shape will be [batch_size]
        source_mask = (domain_labels == 0.0).bool()

        # Task loss (source only, this allows for non-autoregressive training later)
        task_loss = torch.tensor(0.0, device=self.device)
        if source_mask.any():
            # Get source samples using unsqueezed mask for proper broadcasting
            # x shape: [batch_size, seq_len, features]
            # source_mask shape: [batch_size]
            x_source = x[source_mask, :, :]
            y_source = y[source_mask]
            static_source = static[source_mask]

            # Calculate task loss on source samples
            y_pred = self(x_source, static_source)
            task_loss = self.mse_criterion(y_pred, y_source)

        # Domain loss (all samples)
        features = self.extract_features(x, static)
        features_rev = grad_reverse(features, self.hparams.lambda_adv)
        domain_preds = self.domain_discriminator(features_rev)

        # Ensure domain labels match prediction shape for loss calculation
        domain_labels = domain_labels.view_as(domain_preds)
        domain_loss = self.domain_criterion(domain_preds, domain_labels)

        # Combined loss
        total_loss = task_loss + self.hparams.domain_loss_weight * domain_loss

        # Log metrics
        self.log_dict({
            "train_loss": total_loss,
        }, prog_bar=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, static = batch["X"], batch["y"].unsqueeze(-1), batch["static"]
        domain_labels = batch["domain_id"]

        # Convert domain labels if needed
        if not isinstance(domain_labels, torch.Tensor):
            domain_labels = torch.tensor(
                domain_labels, device=self.device).float()

        # Forward pass
        y_pred = self(x, static)

        # Task loss
        task_loss = self.mse_criterion(y_pred, y)

        # Domain predictions
        features = self.extract_features(x, static)
        domain_preds = self.domain_discriminator(features)

        # Domain accuracy
        domain_acc = ((domain_preds > 0.5).float() ==
                      domain_labels).float().mean()

        # Log metrics
        self.log_dict({
            "val_loss": task_loss,
            "val_domain_acc": domain_acc
        }, prog_bar=False)

        return task_loss

    def test_step(self, batch, batch_idx):
        """Test step focused only on prediction performance, not domain adaptation.
        Matches format expected by TSForecastEvaluator."""
        x, y, static = batch["X"], batch["y"], batch["static"]

        # Get predictions
        y_hat = self(x, static)

        # Store predictions, observations and gauge IDs for evaluation
        output = {
            # Remove last dimension for evaluator
            "predictions": y_hat.squeeze(-1),
            # Remove last dimension for evaluator
            "observations": y.squeeze(-1),
            "basin_ids": batch[self.group_identifier]  # Keep track of basins
        }

        # Log MSE loss for monitoring
        test_loss = self.mse_criterion(y_hat, y.unsqueeze(-1))
        self.log("test_loss", test_loss)

        # Store output for later collection
        self.test_outputs.append(output)

        return output

    def on_test_epoch_start(self) -> None:
        """Initialize storage for test results"""
        self.test_outputs = []

    def on_test_epoch_end(self) -> None:
        """Aggregate test results for evaluation"""
        if not self.test_outputs:
            print("Warning: No test outputs collected")
            return

        # Collect all outputs
        self.test_results = {
            "predictions": torch.cat([x["predictions"] for x in self.test_outputs]),
            "observations": torch.cat([x["observations"] for x in self.test_outputs]),
            "basin_ids": [bid for x in self.test_outputs for bid in x["basin_ids"]],
        }

        # Don't clear test_outputs until we've successfully created test_results
        self.test_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
