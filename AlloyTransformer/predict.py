import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import json
from typing import OrderedDict
from trainer import AlloyTransformerLightning  # Import the Lightning module
from dataloader import LM_Dataset, collate_fn # Import your custom Dataset and collate_fn

class Predictor:
    def __init__(self, model_path: str, evaluation_path: str, configs: dict, save_dir: str):
        self.model_path = model_path
        self.evaluation_path = evaluation_path
        self.configs = configs
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_df = None
        self.predictions = None
        self.targets = None

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created save directory: {self.save_dir}")

        print("Loading model...")
        self.model = self.load_lightning_model(self.model_path, self.configs)
        self.model.to(self.device)

        # Ensure model is properly set up
        if hasattr(self.model, 'model') and self.model.model is None:
            print("Running setup for the model in __init__...")
            self.model.setup('test')

        print("Setting up test dataset and loader...")
        self.test_dataset = LM_Dataset(self.evaluation_path, has_targets=False)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn
        )
        print("Predictor initialized.")

    @staticmethod
    def load_lightning_model(checkpoint_path, config=None):
        """
        Load a PyTorch Lightning model from checkpoint with improved handling of different architectures
        """
        # Add safe globals to handle PyTorch 2.6 security changes
        import torch.serialization
        try:
            import torch.torch_version
            torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
        except (ImportError, AttributeError):
            pass  # Skip if not available in this PyTorch version
        
        if checkpoint_path.endswith('.ckpt'):
            # Load from Lightning checkpoint
            try:
                model = AlloyTransformerLightning.load_from_checkpoint(
                    checkpoint_path, 
                    weights_only=False,
                    map_location='cpu'
                )
                print(f"Successfully loaded Lightning checkpoint from {checkpoint_path}")
            except Exception as e:
                print(f"Error loading Lightning checkpoint: {str(e)}")
                # Try without weights_only parameter for older PyTorch versions
                try:
                    model = AlloyTransformerLightning.load_from_checkpoint(
                        checkpoint_path,
                        map_location='cpu'
                    )
                    print("Successfully loaded Lightning checkpoint (fallback method)")
                except Exception as e2:
                    print(f"Failed to load Lightning checkpoint: {str(e2)}")
                    raise e2
        else:
            # Load from state dict
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                print(f"Successfully loaded checkpoint with weights_only=False")
            except TypeError:
                # weights_only parameter not supported in older PyTorch versions
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print(f"Successfully loaded checkpoint (older PyTorch version)")
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                raise e
        
            if config is None:
                raise ValueError("Config must be provided when loading from state dict")
                
            print(f"Creating model with config: hidden_dim={config.get('d_model', 'N/A')}, " 
                  f"num_layers={config.get('num_transformer_layers', 'N/A')}, "
                  f"num_heads={config.get('num_head', 'N/A')}")
            
            model = AlloyTransformerLightning(config)
            
            # Call setup to initialize the inner model
            model.setup('test')
            
            # Extract model state dict
            model_state_dict = None
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model_state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    model_state_dict = checkpoint['state_dict']
                else:
                    # Try direct loading as a state dict
                    model_state_dict = checkpoint
            else:
                model_state_dict = checkpoint
            
            # Try to load the state dict with different strategies
            try:
                # First attempt: direct loading
                model.load_state_dict(model_state_dict)
                print("Successfully loaded model state dict directly.")
            except RuntimeError as e:
                print(f"Direct loading failed: {e}")
                
                # Second attempt: try loading into the inner model
                if hasattr(model, 'model') and model.model is not None:
                    try:
                        # Check if state dict has model prefix
                        if any(key.startswith('model.') for key in model_state_dict.keys()):
                            model_inner_state_dict = {
                                k.replace('model.', ''): v for k, v in model_state_dict.items() 
                                if k.startswith('model.')
                            }
                            model.model.load_state_dict(model_inner_state_dict)
                            print("Successfully loaded inner model state dict.")
                            
                            # Load remaining parameters
                            non_model_state = {
                                k: v for k, v in model_state_dict.items() 
                                if not k.startswith('model.')
                            }
                            if non_model_state:
                                model.load_state_dict(non_model_state, strict=False)
                                print("Loaded non-model parameters.")
                        else:
                            # Try loading directly into inner model
                            model.model.load_state_dict(model_state_dict, strict=False)
                            print("Loaded state dict into inner model with strict=False.")
                    except RuntimeError as e2:
                        print(f"Inner model loading failed: {e2}")
                        print("Attempting to load with strict=False as last resort...")
                        model.load_state_dict(model_state_dict, strict=False)
                        print("Loaded state dict with strict=False. Some parameters may be missing.")
                else:
                    # Last resort: try loading with strict=False
                    print("No inner model found. Attempting to load with strict=False...")
                    model.load_state_dict(model_state_dict, strict=False)
                    print("Loaded state dict with strict=False. Some parameters may be missing.")
        
        # Set model to evaluation mode
        model.eval()
        return model

    def predict(self):
        """Run model predictions on test set and store results in a DataFrame."""
        if self.model is None or self.test_loader is None:
            print("Model or test loader not initialized. Cannot predict.")
            return None

        self.model.eval()
        all_predictions = []
        all_targets = []
        
        print("Starting predictions...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # Handle different batch formats
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    # If batch is just inputs without targets
                    inputs = batch
                    targets = None
                
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)

                try:
                    predictions = self.model(inputs)
                    
                    # Handle different prediction shapes
                    if predictions.ndim > 1:
                        if predictions.shape[-1] == 1:
                            predictions = predictions.squeeze(-1)
                        elif predictions.ndim == 2:
                            # For classification, might need argmax
                            # For regression with multiple outputs, might need different handling
                            pass
                    
                    # Handle targets if they exist
                    if targets is not None:
                        if targets.ndim > 1 and targets.shape[-1] == 1:
                            targets = targets.squeeze(-1)
                        
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        # If no targets, create dummy targets or handle appropriately
                        all_targets.extend([None] * len(predictions))
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
        
        if not all_predictions:
            print("No predictions were generated!")
            return None
            
        self.predictions = np.array(all_predictions)
        if all_targets[0] is not None:
            self.targets = np.array(all_targets)
        else:
            self.targets = None
        
        values = pd.read_csv(self.evaluation_path)

        # Create DataFrame
        try:
            if self.targets is not None:
                self.results_df = pd.DataFrame({
                    'targets': self.targets,
                    'predictions': self.predictions
                })
            else:
                self.results_df = pd.DataFrame({
                    "System": values["System_Name"],
                    'predictions': self.predictions
                })
            print(f"Predictions compiled into DataFrame with shape: {self.results_df.shape}")
        except ValueError as e:
            print(f"Error creating DataFrame: {e}")
            if self.targets is not None:
                print(f"Targets shape: {self.targets.shape}, Predictions shape: {self.predictions.shape}")
            else:
                print(f"Predictions shape: {self.predictions.shape}")
            self.results_df = pd.DataFrame()

    def __call__(self):
        """
        Performs prediction and saves the results to a CSV file.
        Returns the DataFrame containing predictions and targets.
        """
        print(f"Running prediction using model from {self.model_path} on data from {self.evaluation_path}")
        
        self.predict()

        if self.results_df is not None and not self.results_df.empty:
            output_filename = "predictions.csv"
            output_path = os.path.join(self.save_dir, output_filename)
            
            try:
                self.results_df.to_csv(output_path, index=False)
                print(f"✅ Predictions successfully saved to: {output_path}")
                print(f"DataFrame info:")
                print(self.results_df.info())
                print(f"Sample of predictions:")
                print(self.results_df.head())
            except Exception as e:
                print(f"Error saving DataFrame to CSV: {e}")
        elif self.results_df is not None and self.results_df.empty:
             print("⚠️ Prediction ran, but the resulting DataFrame is empty. Check data or prediction process.")
        else:
            print("⚠️ Prediction did not generate a DataFrame. Results not saved.")
            
        return self.results_df

# Example Usage
if __name__ == '__main__':
    checkpoint_dir = 'checkpoints/AlloyTransformer_L1_v0.0.1'
    config_path = os.path.join(checkpoint_dir, 'config.json')
    model_path = os.path.join(checkpoint_dir, 'final_model.pt')

    try:
        with open(config_path, 'r') as config_file:
            configs = json.load(config_file)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        configs = {}  # Provide default config or handle appropriately

    pred = Predictor(
        model_path=model_path,
        evaluation_path="Data/Component_Stratified_Split_Based_on_Augmentation/element_combinations_with_ratios.csv",
        configs=configs,
        save_dir="predictions"
    )
    
    # Run predictions
    results = pred()
    
    if results is not None:
        print(f"Prediction completed. Results shape: {results.shape}")
    else:
        print("Prediction failed or returned no results.")