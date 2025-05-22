import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics.regression import (
    MeanAbsolutePercentageError,
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)

from alloytransformer import AlloyTransformer
from dataloader import LM_Dataset, collate_fn


class AlloyTransformerLightning(pl.LightningModule):
    def __init__(self, config):
        super(AlloyTransformerLightning, self).__init__()

        self.save_hyperparameters(config)
        self.model = None
        self.feature_dim = None

        self.criterion = nn.L1Loss()
        self.val_mse = MeanSquaredError()
        self.val_r2 = R2Score()
        self.val_mape = MeanAbsolutePercentageError()

        self.test_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_r2 = R2Score()
        self.test_mape = MeanAbsolutePercentageError()

    def setup(self, stage=None):
        """Setup is called once after data is available"""
        if self.model is None:
            train_dataset = LM_Dataset(self.hparams.train_path)
            sample_data = train_dataset[0]

            # Add better error handling or validation
            if not isinstance(sample_data, tuple) or len(sample_data) < 1:
                raise ValueError(
                    "Dataset sample should return a tuple with at least input tensor"
                )

            if sample_data[0].dim() != 2:
                raise ValueError(
                    f"Expected input tensor with shape (seq_len, features), got {sample_data[0].shape}"
                )

            self.feature_dim = sample_data[0].shape[1]

            self.model = AlloyTransformer(
                feature_dim=self.feature_dim,
                d_model=self.hparams.d_model,
                num_head=self.hparams.num_head,  
                num_transformer_layers=self.hparams.num_transformer_layers,
                num_regression_head_layers=self.hparams.num_regression_head_layers,
                dropout=self.hparams.dropout,
                num_positions=self.hparams.num_positions,  
                dim_feedforward=self.hparams.dim_feedforward,
                use_property_focus=self.hparams.use_property_focus
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor)->torch.Tensor:
        inputs, targets = batch

        targets = targets.squeeze(-1)

        predictions = self(inputs)
        predictions = predictions.squeeze()

        loss = self.criterion(predictions, targets)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor)-> torch.Tensor:
        inputs, targets = batch
        targets = targets.squeeze(-1)

        predictions = self(inputs)
        predictions = predictions.squeeze()

        loss = self.criterion(predictions, targets)

        self.val_mse(predictions, targets)
        self.val_r2(predictions, targets)
        self.val_mape(predictions, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mape", self.val_mape, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def test_step(self, batch: torch.Tensor)->torch.Tensor:
        inputs, targets = batch
        targets = targets.squeeze(-1)

        predictions = self(inputs)
        predictions = predictions.squeeze()

        loss = self.criterion(predictions, targets)

        self.test_mae(predictions, targets)
        self.test_mse(predictions, targets)
        self.test_r2(predictions, targets)
        self.test_mape(predictions, targets)

        # Added logging for test metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_mae", self.test_mae, on_step=False, on_epoch=True)
        self.log("test_mse", self.test_mse, on_step=False, on_epoch=True)
        self.log("test_r2", self.test_r2, on_step=False, on_epoch=True)
        self.log("test_mape", self.test_mape, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        metrics = self.trainer.callback_metrics
        print("\nTest Results:")
        print(f"Loss: {metrics['test_loss']:.4f}")
        print(f"MAE: {metrics['test_mae']:.4f}")
        print(f"MSE: {metrics['test_mse']:.4f}")
        print(f"R²: {metrics['test_r2']:.4f}")
        print(f"MAPE: {metrics['test_mape']:.4f}%")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        train_dataset = LM_Dataset(self.hparams.train_path)
        return DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        val_dataset = LM_Dataset(self.hparams.valid_path)
        return DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        test_dataset = LM_Dataset(self.hparams.test_path)
        return DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
        )
