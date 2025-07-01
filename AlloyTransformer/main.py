import os
import json
import torch

import pytorch_lightning as pl

from torchinfo import summary
from datetime import datetime

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from configs import ConfigLoader, ModelConfig
from trainer import AlloyTransformerLightning
from helper import create_model_loader, save_config, save_final_model, get_next_version


from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


class CustomRichProgressBar(RichProgressBar):
    """Custom RichProgressBar that removes v_num from display and has custom theming"""
    
    def __init__(self, **kwargs):
        # Set up the custom theme
        theme = RichProgressBarTheme(
            description="bold magenta",
            progress_bar="#5AC8FA",             # Light blue
            progress_bar_finished="#32CD32",     # Lime green
            progress_bar_pulse="#FF69B4",        # Hot pink
            batch_progress="#FFD700",            # Gold
            time="grey70",
            processing_speed="grey70",
            metrics="white",
            metrics_text_delimiter="\n"
        )
        
        # Initialize with custom theme
        super().__init__(theme=theme, **kwargs)
    
    def get_metrics(self, trainer, pl_module):
        # Get default metrics
        items = super().get_metrics(trainer, pl_module)
        # Remove v_num if it exists
        items.pop("v_num", None)
        return items


def setup_checkpoints(checkpoint_dir, patience):
    """Set up checkpoint callbacks - only saving best and last"""
    best_val_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='best-val-{epoch:03d}-{val_loss:.4f}',
        mode='min',
        save_last=True,
        verbose=True
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=patience,
        mode='min',
        verbose=True
    )
    
    # Custom rich progress bar without v_num and with custom theming
    custom_progress_bar = CustomRichProgressBar()
    
    return [
        best_val_checkpoint,
        lr_monitor,
        early_stopping,
        custom_progress_bar
    ], best_val_checkpoint

def main(configs: ModelConfig):
    """Main training function"""
    
    # Set precision for better performance
    torch.set_float32_matmul_precision("highest")

    # Set random seed for reproducibility
    pl.seed_everything(configs.seed)
    
    # Create timestamp and version info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version_prefix = f"v{configs.model_version}"
    
    # Setup directories
    logs_dir = 'TensorBoard'
    
    # Get next version number
    version = get_next_version(
        logs_dir=logs_dir, 
        model_name=configs.model_name, 
        model_version_prefix=model_version_prefix
    )
    
    run_name = f"{configs.model_name}_{version}"
    checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save configuration
    save_config(configs, checkpoint_dir)

    # Initialize model
    print("Initializing AlloyTransformer model...")
    model = AlloyTransformerLightning(configs.as_dict())
    
    # Setup callbacks
    callbacks, best_val_checkpoint = setup_checkpoints(checkpoint_dir, configs.patience)

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=logs_dir,
        name=configs.model_name,
        version=version,  # Use your generated version instead of None
        default_hp_metric=False
    )
    
    # Determine device configuration
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=configs.num_epochs,
        accelerator='auto',  # Automatically select CPU/GPU
        devices=gpus if gpus > 0 else 1,  # Use available GPUs or 1 CPU
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=configs.gradient_clip,
        deterministic=True,
        log_every_n_steps=1,
        # Additional useful settings
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Log hyperparameters
    logger.log_hyperparams(configs.as_dict())
    
    # Print training info
    print(f"\n{'='*60}")
    print(f"Starting training run: {run_name}")
    print(f"TensorBoard logs: {os.path.join(logs_dir, configs.model_name, version)}")
    print(f"Checkpoints: {checkpoint_dir}")
    
    if gpus > 0:
        print(f"Training for {configs.num_epochs} epochs on {gpus} GPU(s)")
        print(f"GPU(s) available: {[torch.cuda.get_device_name(i) for i in range(gpus)]}")
    else:
        print(f"Training for {configs.num_epochs} epochs on CPU")
    
    print(f"Early stopping patience: {configs.patience} epochs")
    print(f"{'='*60}\n")
    
    # Start training
    try:
        trainer.fit(model)
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
        
    # Get best checkpoint
    best_checkpoint_path = best_val_checkpoint.best_model_path
    print(f"\nBest checkpoint: {best_checkpoint_path}")
    
    # Test evaluation
    test_results = None
    if hasattr(configs, 'test_path') and configs.test_path:
        print("\nEvaluating on test set...")
        try:
            test_results = trainer.test(model, ckpt_path='best')
            
            # Save test results
            test_results_path = os.path.join(checkpoint_dir, "test_results.json")
            with open(test_results_path, 'w') as f:
                json.dump(test_results, f, indent=4)
            print(f"Test results saved to: {test_results_path}")
            
        except Exception as e:
            print(f"Error during testing: {e}")
    else:
        print("No test path specified, skipping test evaluation")
    
    # Save final model
    print("\nSaving final model...")
    final_model_path = save_final_model(
        model=model,
        config=configs,
        trainer=trainer,
        best_checkpoint=best_val_checkpoint,
        timestamp=timestamp,
        checkpoint_dir=checkpoint_dir,
        test_results=test_results
    )
    
    # Generate model summary
    print("Generating model summary...")
    try:
        # Ensure model is set up
        model.setup()
        base_model = model.model  # Access the underlying transformer model
        
        if base_model is not None:
            # Create a sample input tensor (batch_size, seq_len, feature_dim)
            # Use the actual feature_dim from the model setup
            feature_dim = model.feature_dim if hasattr(model, 'feature_dim') else configs.feature_dim
            sample_input = torch.zeros((1, 5, feature_dim), dtype=torch.float32)
            
            model_summary = summary(
                base_model, 
                input_data=sample_input, 
                depth=4, 
                verbose=0
            )
            
            summary_path = os.path.join(checkpoint_dir, "model_summary.txt")
            with open(summary_path, 'w') as f:
                print(model_summary, file=f)
            print(f"Model summary saved to: {summary_path}")
            
        else:
            print("Warning: Base model not initialized, skipping summary generation")
            
    except Exception as e:
        print(f"Error generating model summary: {e}")
    
    # Create model loader
    print("Creating model loader...")
    try:
        create_model_loader(checkpoint_dir, configs, final_model_path)
        print("Model loader created successfully")
    except Exception as e:
        print(f"Error creating model loader: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Final model saved to: {final_model_path}")
    print(f"All training artifacts saved in: {checkpoint_dir}")
    print(f"TensorBoard logs: {os.path.join(logs_dir, configs.model_name, version)}")
    
    if test_results:
        print(f"\nFinal Test Results:")
        for result_dict in test_results:
            for key, value in result_dict.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    print(f"{'='*60}")
    
    # Instructions for TensorBoard
    print(f"\nTo view training progress in TensorBoard, run:")
    print(f"tensorboard --logdir={logs_dir}")
    
    return {
        'final_model_path': final_model_path,
        'checkpoint_dir': checkpoint_dir,
        'test_results': test_results,
        'run_name': run_name
    }


if __name__ == '__main__':
    try:
        # Load configuration
        config = ConfigLoader.load('configs/configs.yml')
        
        # Run main training
        results = main(configs=config)
        
        print("\nTraining pipeline completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
    except Exception as e:
        print(f"Training pipeline failed: {e}")
        raise