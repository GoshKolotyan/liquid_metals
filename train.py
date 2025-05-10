import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.regression import MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError, R2Score
from tokenizer import LM_Tokenizer
from dataloader import LM_Dataset, collate_fn
from model import ChemicalTransformer


class ChemicalTransformerLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Initialize model parameters (will be set in setup)
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
            self.feature_dim = sample_data[0].shape[1]  # Assuming shape is (seq_len, features)
            
            self.model = ChemicalTransformer(
                feature_dim=self.feature_dim,
                hidden_dim=self.hparams.hidden_dim,
                num_heads=self.hparams.num_heads,
                num_layers=self.hparams.num_layers,
                dropout=self.hparams.dropout,
                max_seq_length=self.hparams.max_seq_length
            )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.squeeze(-1)
        
        predictions = self(inputs)
        predictions = predictions.squeeze()
        
        loss = self.criterion(predictions, targets)
        
        # self.train_mae(predictions, targets)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.squeeze(-1)
        
        predictions = self(inputs)
        predictions = predictions.squeeze()
        
        loss = self.criterion(predictions, targets)
        
        self.val_mse(predictions, targets)
        self.val_r2(predictions, targets)
        self.val_mape(predictions, targets)
        
        rmse = torch.sqrt(self.val_mse(predictions, targets))
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mse', self.val_mse, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_rmse', rmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_r2', self.val_r2, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mape', self.val_mape, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.squeeze(-1)
        
        predictions = self(inputs)
        predictions = predictions.squeeze()
        
        loss = self.criterion(predictions, targets)
        
        self.test_mae(predictions, targets)
        self.test_mse(predictions, targets)
        self.test_r2(predictions, targets)
        self.test_mape(predictions, targets)
        
        rmse = torch.sqrt(self.test_mse(predictions, targets))
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_mae', self.test_mae, on_step=False, on_epoch=True)
        self.log('test_mse', self.test_mse, on_step=False, on_epoch=True)
        self.log('test_rmse', rmse, on_step=False, on_epoch=True)
        self.log('test_r2', self.test_r2, on_step=False, on_epoch=True)
        self.log('test_mape', self.test_mape, on_step=False, on_epoch=True)
        
        return loss
    
    def on_test_epoch_end(self):
        metrics = self.trainer.callback_metrics
        print("\nTest Results:")
        print(f"Loss: {metrics['test_loss']:.4f}")
        print(f"MAE: {metrics['test_mae']:.4f}")
        print(f"MSE: {metrics['test_mse']:.4f}")
        print(f"RMSE: {metrics['test_rmse']:.4f}")
        print(f"R²: {metrics['test_r2']:.4f}")
        print(f"MAPE: {metrics['test_mape']:.4f}%")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def train_dataloader(self):
        train_dataset = LM_Dataset(self.hparams.train_path)
        return DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers
        )
    
    def val_dataloader(self):
        val_dataset = LM_Dataset(self.hparams.valid_path)
        return DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers
        )
    
    def test_dataloader(self):
        test_dataset = LM_Dataset(self.hparams.test_path)
        return DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers
        )


def main():
    # Set float32 matmul precision for better performance on GPUs with Tensor Cores
    torch.set_float32_matmul_precision('medium')
    
    config = {
        'train_path': './Data/Custom_Output/train.csv',
        'valid_path': './Data/Custom_Output/valid.csv',
        'test_path': './Data/Custom_Output/test.csv',
        
        'seed': 42,
        'learning_rate': 0.0005,
        'num_epochs': 250,
        'batch_size': 256,
        'num_workers': 8,
        
        'hidden_dim': 128,
        'num_heads': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'max_seq_length': 512,
        
        'patience': 10,
        'gradient_clip': 1.0,
    }
    
    # Set seed for reproducibility
    pl.seed_everything(config['seed'])
    
    model = ChemicalTransformerLightning(config)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='chemical-transformer-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['patience'],
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    logger = TensorBoardLogger('logs', name='chemical_transformer')
    
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        accelerator='auto',
        devices=1,
        logger=logger,
        # callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        gradient_clip_val=config['gradient_clip'],
        deterministic=True,
        log_every_n_steps=1,
    )
    
    print(f"Training for {config['num_epochs']} epochs:")
    trainer.fit(model)
    
    if config.get('test_path'):
        print("\nEvaluating on test set...")
        trainer.test(model, ckpt_path='best')
    
    torch.save(model.state_dict(), 'final_chemical_transformer_lightning_2.pth')
    print(f"\nFinal model saved to: final_chemical_transformer_lightning.pth")


if __name__ == '__main__':
    main()