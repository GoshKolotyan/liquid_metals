import os
import json
import torch

import pytorch_lightning as pl

from torchinfo import summary
from datetime import datetime

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from configs import ConfigLoader, ModelConfig
from trainer import AlloyTransformerLightning
from helper import create_model_loader, save_config, save_final_model, get_next_version

def setup_checkpoints(checkpoint_dir, patience):
    """Set up checkpoint callbacks - only saving best and last"""
    best_val_checkpoint = ModelCheckpoint(monitor='val_loss',dirpath=checkpoint_dir,
                                          filename='best-val-{epoch:03d}-{val_loss:.4f}',
                                          mode='min',save_last=True,  verbose=True
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


def main(configs:ModelConfig):

    torch.set_float32_matmul_precision("highest")

    pl.seed_everything(configs.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version_prefix = f"v{configs.model_version}"
    

    logs_dir = 'log'

    version = get_next_version(logs_dir=logs_dir, 
                               model_name=configs.model_name, 
                               model_version_prefix=model_version_prefix)
    run_name = f"{configs.model_name}_{version}"
    checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    save_config(configs, checkpoint_dir)

    model = AlloyTransformerLightning(configs.as_dict())
    callbacks, best_val_checkpoint = setup_checkpoints(checkpoint_dir, configs.patience)

    logger = TensorBoardLogger(
        save_dir=logs_dir,
        name=configs.model_name,
        version=version,  # Use our custom versioning format
        default_hp_metric=False
    )
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    trainer = pl.Trainer(
        max_epochs=configs.num_epochs,
        accelerator='auto',  # Automatically select CPU/GPU
        devices=gpus if gpus > 0 else 1,  # Use available GPUs or 1 CPU
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=configs.gradient_clip,
        deterministic=True,
        log_every_n_steps=1,
    )
    logger.log_hyperparams(configs.as_dict())
    
    print(f"Starting training run: {run_name}")
    print(f"TensorBoard logs will be saved to: {os.path.join(logs_dir, configs.model_name, version)}")
    print(f"Training for {configs.num_epochs} epochs on {gpus} GPU(s)" if gpus > 0 else f"Training for {configs.num_epochs} epochs on CPU")
    trainer.fit(model)
        
    best_checkpoint_path = best_val_checkpoint.best_model_path
    print(f"Best checkpoint: {best_checkpoint_path}")
    
    test_results = None
    if configs.test_path:
        print("\nEvaluating on test set...")
        test_results = trainer.test(model, ckpt_path='best')
        
        with open(os.path.join(checkpoint_dir, "test_results.json"), 'w') as f:
                json.dump(test_results, f, indent=4)  
    final_model_path = save_final_model(
        model=model,
        config=configs,
        trainer=trainer,
        best_checkpoint=best_val_checkpoint,
        timestamp=timestamp,
        checkpoint_dir=checkpoint_dir,
        test_results=test_results
    )
    try:
        # Get a sample input to use with summary
        model.setup()  # Ensure model is set up
        base_model = model.model  # Access the underlying transformer model
        
        if base_model is not None:
            # Create a sample input tensor (batch_size, seq_len, feature_dim)
            sample_input = torch.zeros((1, 5, configs.feature_dim), dtype=torch.float32)
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
    config = ConfigLoader.load('configs/configs.yml')
    main(configs=config)