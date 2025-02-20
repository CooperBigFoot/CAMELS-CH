import torch.nn as nn
import torch
from torch.optim import Adam
from .models import TSMixer
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

    def __init__(self, config, lambda_adv=1.0, domain_loss_weight=1.0):
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

    def extract_features(self, x, static):
        features = self.model.backbone(x, static)
        return features.flatten(start_dim=1)

    def training_step(self, batch, batch_idx):
        x, y, static = batch["X"], batch["y"].unsqueeze(-1), batch["static"]
        domain_labels = batch["domain_id"].long()

        # Task loss (source only)
        source_mask = domain_labels == 0
        task_loss = self.mse_criterion(
            self(x[source_mask], static[source_mask]),
            y[source_mask]
        ) if source_mask.any() else 0.0

        # Domain loss (all samples)
        features = self.extract_features(x, static)
        features_rev = grad_reverse(features, self.hparams.lambda_adv)
        domain_logits = self.domain_discriminator(features_rev)
        domain_loss = self.domain_criterion(domain_logits, domain_labels)

        # Combined loss
        total_loss = task_loss + self.hparams.domain_loss_weight * domain_loss

        self.log_dict({
            "train_loss": total_loss,
            "task_loss": task_loss,
            "domain_loss": domain_loss
        }, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        # Original task metrics
        x, y, static = batch["X"], batch["y"], batch["static"]
        y_hat = self(x, static)
        task_loss = self.mse_criterion(y_hat, y.unsqueeze(-1))

        # Domain confusion metrics
        features = self.extract_features(x, static)
        domain_preds = self.domain_discriminator(features).argmax(dim=1)
        domain_acc = (domain_preds == batch["domain_id"]).float().mean()

        self.log_dict({
            "val_loss": task_loss,
            "val_domain_acc": domain_acc
        }, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.config.learning_rate)
