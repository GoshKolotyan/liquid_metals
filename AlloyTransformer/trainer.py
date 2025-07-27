import torch
import numpy as np
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from io import BytesIO
from scipy import stats
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

        self.criterion = nn.L1Loss()  # Default loss function
        self.val_mse = MeanSquaredError()
        self.val_r2 = R2Score()
        self.val_mape = MeanAbsolutePercentageError()

        self.test_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_r2 = R2Score()
        self.test_mape = MeanAbsolutePercentageError()
        
        # Visualization settings
        self.plot_every_n_epochs = getattr(config, 'plot_every_n_epochs', 2)
        self.max_plot_samples = getattr(config, 'max_plot_samples', 1500)  # Limit samples for performance
        self.max_line_plot_samples = getattr(config, 'max_line_plot_samples', 250)  # For line plot performance
        
        self.validation_predictions = []
        self.validation_targets = []

    def setup(self, stage=None):
        """Setup is called once after data is available"""
        if self.model is None:
            train_dataset = LM_Dataset(self.hparams.train_path)
            sample_data = train_dataset[0]

            #better error handling or validation
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
                use_property_focus=self.hparams.use_property_focus,
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

    def validation_step(self, batch: torch.Tensor, batch_idx: int)-> torch.Tensor:
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

        # Store predictions and targets for plotting (Lightning v2.0+ approach)
        if self.should_create_plot():
            self.validation_predictions.append(predictions.detach().cpu())
            self.validation_targets.append(targets.detach().cpu())
        
        return loss

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch - Lightning v2.0+ approach"""
        if not self.should_create_plot() or len(self.validation_predictions) == 0:
            self.validation_predictions.clear()
            self.validation_targets.clear()
            return
            
        # Concatenate all stored predictions and targets
        predictions = torch.cat(self.validation_predictions, dim=0)
        targets = torch.cat(self.validation_targets, dim=0)
        
        # Create and log both plots
        self.create_and_log_validation_plots(targets.numpy(), predictions.numpy())
        
        # Clear stored data
        self.validation_predictions.clear()
        self.validation_targets.clear()

    def should_create_plot(self):
        """Determine if we should create a plot this epoch"""
        return (self.current_epoch % self.plot_every_n_epochs == 0 or 
                self.current_epoch == 0)  # Always plot first epoch

    def create_and_log_validation_plots(self, targets, predictions):
        """Create and log both scatterplot and line plot to TensorBoard"""
        try:
            # Create scatterplot
            self.create_and_log_scatterplot(targets, predictions)
            
            # Create line plot
            self.create_and_log_line_plot(targets, predictions)
            
        except Exception as e:
            print(f"Warning: Could not create validation plots: {e}")
            plt.close('all')  # Clean up any remaining figures

    def create_and_log_scatterplot(self, targets, predictions):
        """Create and log scatterplot to TensorBoard"""
        try:
            # Subsample if too many points for scatter plot
            scatter_targets = targets
            scatter_predictions = predictions
            if len(predictions) > self.max_plot_samples:
                indices = np.random.choice(len(predictions), self.max_plot_samples, replace=False)
                scatter_targets = targets[indices]
                scatter_predictions = predictions[indices]
            
            # Create figure
            plt.style.use('default')  # Reset style
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
            
            # Create scatterplot with some styling
            scatter = ax.scatter(scatter_targets, scatter_predictions, alpha=0.6, s=30, c='blue', edgecolors='none')
            
            # Add perfect prediction line
            min_val = min(np.min(scatter_targets), np.min(scatter_predictions))
            max_val = max(np.max(scatter_targets), np.max(scatter_predictions))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                   label='Perfect Prediction', alpha=0.8)
            
            # Calculate metrics for display
            r2_score = self.val_r2.compute().item()
            mse_score = self.val_mse.compute().item()
            
            # Styling
            ax.set_xlabel('True Values', fontsize=14, fontweight='bold')
            ax.set_ylabel('Predicted Values', fontsize=14, fontweight='bold')
            ax.set_title(f'Validation Scatter Plot - Epoch {self.current_epoch}\n'
                        f'R² = {r2_score:.4f} | MSE = {mse_score:.4f}', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Make it square and add some padding
            ax.set_aspect('equal', adjustable='box')
            margin = (max_val - min_val) * 0.05
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)
            
            # Add text box with additional info
            textstr = f'Samples: {len(scatter_targets)}\nEpoch: {self.current_epoch}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # Log to TensorBoard
            self._log_plot_to_tensorboard(fig, 'validation/scatter_plot')
            
            # Clean up
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not create validation scatter plot: {e}")
            plt.close('all')

    def create_and_log_line_plot(self, targets, predictions):
        """Create and log line plot showing real vs predicted values over sample indices"""
        try:
            # Subsample for line plot if too many points (for readability)
            line_targets = targets
            line_predictions = predictions
            if len(predictions) > self.max_line_plot_samples:
                indices = np.random.choice(len(predictions), self.max_line_plot_samples, replace=False)
                indices = np.sort(indices)  # Sort to maintain order for line plot
                line_targets = targets[indices]
                line_predictions = predictions[indices]
            
            # Create x-axis values (sample indices)
            x_values = np.arange(len(line_targets))
            
            # Calculate metrics
            mae = np.mean(np.abs(line_targets - line_predictions))
            rmse = np.sqrt(np.mean((line_targets - line_predictions)**2))
            
            # Calculate correlation coefficient for R²
            if len(line_targets) > 1:
                correlation = stats.pearsonr(line_targets, line_predictions)[0]
                r2 = correlation**2 if not np.isnan(correlation) else 0.0
            else:
                r2 = 0.0
            
            # Create figure
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
            
            # Plot real values and predictions
            ax.plot(x_values, line_targets, 'o-', linewidth=2, markersize=6, 
                    color='blue', alpha=0.7, label='Real Values')
            ax.plot(x_values, line_predictions, 's-', linewidth=2, markersize=6, 
                    color='red', alpha=0.7, label='Predictions')
            
            ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('Property Value', fontsize=12, fontweight='bold')
            ax.set_title(f'Validation Line Plot - Epoch {self.current_epoch}\n'
                        f'R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            
            # Add text box with additional info
            textstr = f'Samples: {len(line_targets)}\nEpoch: {self.current_epoch}'
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # Log to TensorBoard
            self._log_plot_to_tensorboard(fig, 'validation/line_plot')
            
            # Clean up
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not create validation line plot: {e}")
            plt.close('all')

    def _log_plot_to_tensorboard(self, fig, tag):
        """Helper method to log matplotlib figure to TensorBoard"""
        try:
            # Convert to image for TensorBoard
            buf = BytesIO()
            plt.figure(fig.number)  # Make sure we're working with the right figure
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
            buf.seek(0)
            
            # Log to TensorBoard
            if hasattr(self.logger, 'experiment'):
                from PIL import Image
                pil_image = Image.open(buf)
                img_array = np.array(pil_image)
                
                # Convert to CHW format for TensorBoard
                if len(img_array.shape) == 3:
                    img_array = np.transpose(img_array, (2, 0, 1))
                
                self.logger.experiment.add_image(
                    tag,
                    img_array,
                    self.current_epoch
                )
            
            # Clean up
            buf.close()
            
        except Exception as e:
            print(f"Warning: Could not log plot to TensorBoard: {e}")

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