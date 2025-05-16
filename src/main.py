import os
import json
import torch

import pytorch_lightning as pl

from torchinfo import summary

from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from configs import ConfigLoader
from train import ChemicalTransformerLightning
from helper import create_model_loader, save_config, save_final_model, get_next_version


def setup_checkpoints(checkpoint_dir, patience):
    """Set up checkpoint callbacks - only saving best and last"""
    # Create best validation checkpoint callback
    best_val_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='best-val-{epoch:03d}-{val_loss:.4f}',
        mode='min',
        save_last=True,  # Also save the last model
        verbose=True
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Early stopping callback (adding this since it was imported but not used)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        verbose=True
    )
    
    return [
        best_val_checkpoint,
        lr_monitor,
        early_stopping
    ], best_val_checkpoint



def main(config):
    # Set precision for matrix multiplications
    torch.set_float32_matmul_precision('highest')
    
    # Set seed for reproducibility
    pl.seed_everything(config.seed)
    
    # Create unique run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version_prefix = f"v{config.model_version}"
    
    # Get the next version number for this model version
    logs_dir = 'logs'
    version = get_next_version(logs_dir, config.model_name, model_version_prefix)
    
    run_name = f"{config.model_name}_{version}_{timestamp}"
    checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, checkpoint_dir)
    
    # Initialize model
    model = ChemicalTransformerLightning(config)
    
    # Setup callbacks and logger
    callbacks, best_val_checkpoint = setup_checkpoints(checkpoint_dir, config.patience)
    
    # Create logger with custom version naming
    logger = TensorBoardLogger(
        save_dir=logs_dir,
        name=config.model_name,
        version=version,  # Use our custom versioning format
        default_hp_metric=False
    )
    
    # Determine number of GPUs to use
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator='auto',  # Automatically select CPU/GPU
        devices=gpus if gpus > 0 else 1,  # Use available GPUs or 1 CPU
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config.gradient_clip,
        deterministic=True,
        log_every_n_steps=1,
    )
    
    # Log hyperparameters
    logger.log_hyperparams(config.as_dict())
    
    # Train model
    print(f"Starting training run: {run_name}")
    print(f"TensorBoard logs will be saved to: {os.path.join(logs_dir, config.model_name, version)}")
    print(f"Training for {config.num_epochs} epochs on {gpus} GPU(s)" if gpus > 0 else f"Training for {config.num_epochs} epochs on CPU")
    trainer.fit(model)
    
    
    # Get best checkpoint path
    best_checkpoint_path = best_val_checkpoint.best_model_path
    print(f"Best checkpoint: {best_checkpoint_path}")
    
    # Test if test path is provided
    test_results = None
    if config.test_path:
        print("\nEvaluating on test set...")
        test_results = trainer.test(model, ckpt_path='best')
        
        with open(os.path.join(checkpoint_dir, "test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=4)
    
    # Save final model
    final_model_path = save_final_model(
        model=model,
        config=config,
        trainer=trainer,
        best_checkpoint=best_val_checkpoint,
        timestamp=timestamp,
        checkpoint_dir=checkpoint_dir,
        test_results=test_results
    )
    
    # Generate model summary - use the actual model, not the Lightning module
    try:
        # Get a sample input to use with summary
        model.setup()  # Ensure model is set up
        base_model = model.model  # Access the underlying transformer model
        
        if base_model is not None:
            # Create a sample input tensor (batch_size, seq_len, feature_dim)
            sample_input = torch.zeros((1, config.max_seq_length, model.feature_dim))
            model_summary = summary(base_model, input_data=sample_input, depth=4, verbose=0)
            
            with open(os.path.join(checkpoint_dir, "model_summary.txt"), 'w') as f:
                print(model_summary, file=f)
        else:
            print("Base model not initialized, skipping summary generation")
    except Exception as e:
        print(f"Error generating model summary: {e}")
    
    # Create model loader
    create_model_loader(checkpoint_dir, config, final_model_path)
    
    print(f"\nAdvanced model saved to: {final_model_path}")
    print(f"All training artifacts saved in: {checkpoint_dir}")


if __name__ == '__main__':
    config = ConfigLoader.load('configs/config.yaml')
    main(config=config)