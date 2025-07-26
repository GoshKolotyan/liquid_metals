import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import OrderedDict
from trainer import AlloyTransformerLightning 
from dataloader import LM_Dataset, collate_fn 
import re
from mpl_toolkits.mplot3d import Axes3D

class PredictorWithPlots:
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

        # Create subdirectories for plots
        self.plots_dir = os.path.join(self.save_dir, "phase_diagrams")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, "ternary"), exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, "quaternary"), exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, "pentanary"), exist_ok=True)

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
            pass  
        
        if checkpoint_path.endswith('.ckpt'):
            try:
                model = AlloyTransformerLightning.load_from_checkpoint(
                    checkpoint_path, 
                    weights_only=False,
                    map_location='cpu'
                )
                print(f"Successfully loaded Lightning checkpoint from {checkpoint_path}")
            except Exception as e:
                print(f"Error loading Lightning checkpoint: {str(e)}")
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

    def parse_composition(self, system_name):
        """Parse composition from system name like 'Ga0.3In0.3Sn0.4' -> (['Ga', 'In', 'Sn'], [0.3, 0.3, 0.4])"""
        # Find all element-ratio pairs
        pattern = r'([A-Z][a-z]*)([0-9]*\.?[0-9]+)'
        matches = re.findall(pattern, system_name)
        
        elements = []
        compositions = []
        
        for element, ratio_str in matches:
            elements.append(element)
            compositions.append(float(ratio_str))
        
        return elements, compositions

    def group_predictions_by_elements(self):
        """Group predictions by unique element combinations"""
        if self.results_df is None:
            return {}
        
        element_groups = {}
        
        for idx, row in self.results_df.iterrows():
            system_name = row['System']
            prediction = row['predictions']
            
            elements, compositions = self.parse_composition(system_name)
            element_key = ''.join(sorted(elements))
            
            if element_key not in element_groups:
                element_groups[element_key] = {
                    'elements': sorted(elements),
                    'data': []
                }
            
            element_groups[element_key]['data'].append({
                'system_name': system_name,
                'elements': elements,
                'compositions': compositions,
                'prediction': prediction
            })
        
        return element_groups

    def plot_ternary_diagram(self, element_group, element_key):
        """Create ternary diagram for 3-component systems"""
        data = element_group['data']
        elements = element_group['elements']
        
        if len(elements) != 3:
            return
        
        # Extract compositions and predictions
        compositions = []
        predictions = []
        
        for item in data:
            # Sort compositions according to sorted element order
            element_order = {elem: i for i, elem in enumerate(sorted(elements))}
            sorted_comp = [0] * 3
            
            for elem, comp in zip(item['elements'], item['compositions']):
                sorted_comp[element_order[elem]] = comp
            
            compositions.append(sorted_comp)
            predictions.append(item['prediction'])
        
        compositions = np.array(compositions)
        predictions = np.array(predictions)
        
        # Create ternary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert to barycentric coordinates for ternary plot
        # For a triangle with vertices at (0,0), (1,0), (0.5, sqrt(3)/2)
        x = 0.5 * (2 * compositions[:, 1] + compositions[:, 2])
        y = (np.sqrt(3) / 2) * compositions[:, 2]
        
        # Create scatter plot with color based on predictions
        scatter = ax.scatter(x, y, c=predictions, cmap='viridis', s=50, alpha=0.7)
        
        # Draw triangle
        triangle_x = [0, 1, 0.5, 0]
        triangle_y = [0, 0, np.sqrt(3)/2, 0]
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)
        
        # Add labels
        ax.text(-0.05, -0.05, elements[0], fontsize=12, ha='center', va='center')  # Bottom left
        ax.text(1.05, -0.05, elements[1], fontsize=12, ha='center', va='center')   # Bottom right  
        ax.text(0.5, np.sqrt(3)/2 + 0.05, elements[2], fontsize=12, ha='center', va='center')  # Top
        
        # Set equal aspect ratio and remove axes
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Predicted Value', rotation=270, labelpad=20)
        
        plt.title(f'Ternary Phase Diagram: {element_key}', fontsize=14, pad=20)
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(self.plots_dir, "ternary", f"{element_key}_ternary.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ternary diagram: {filename}")

    def plot_quaternary_diagram(self, element_group, element_key):
        """Create plots for 4-component systems using multiple 2D projections"""
        data = element_group['data']
        elements = element_group['elements']
        
        if len(elements) != 4:
            return
        
        # Extract compositions and predictions
        compositions = []
        predictions = []
        
        for item in data:
            element_order = {elem: i for i, elem in enumerate(sorted(elements))}
            sorted_comp = [0] * 4
            
            for elem, comp in zip(item['elements'], item['compositions']):
                sorted_comp[element_order[elem]] = comp
            
            compositions.append(sorted_comp)
            predictions.append(item['prediction'])
        
        compositions = np.array(compositions)
        predictions = np.array(predictions)
        
        # Create subplots for different 2D projections
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Define pairs for 2D projections
        pairs = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3)
        ]
        
        for idx, (i, j) in enumerate(pairs):
            ax = axes[idx]
            
            scatter = ax.scatter(compositions[:, i], compositions[:, j], 
                               c=predictions, cmap='viridis', s=30, alpha=0.7)
            
            ax.set_xlabel(f'{elements[i]} Fraction')
            ax.set_ylabel(f'{elements[j]} Fraction')
            ax.set_title(f'{elements[i]} vs {elements[j]}')
            ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=axes, shrink=0.8, aspect=30)
        cbar.set_label('Predicted Value', rotation=270, labelpad=20)
        
        plt.suptitle(f'Quaternary System: {element_key}', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(self.plots_dir, "quaternary", f"{element_key}_quaternary.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved quaternary diagram: {filename}")

    def plot_pentanary_diagram(self, element_group, element_key):
        """Create plots for 5-component systems using heatmaps and parallel coordinates"""
        data = element_group['data']
        elements = element_group['elements']
        
        if len(elements) != 5:
            return
        
        # Extract compositions and predictions
        compositions = []
        predictions = []
        system_names = []
        
        for item in data:
            element_order = {elem: i for i, elem in enumerate(sorted(elements))}
            sorted_comp = [0] * 5
            
            for elem, comp in zip(item['elements'], item['compositions']):
                sorted_comp[element_order[elem]] = comp
            
            compositions.append(sorted_comp)
            predictions.append(item['prediction'])
            system_names.append(item['system_name'])
        
        compositions = np.array(compositions)
        predictions = np.array(predictions)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Parallel coordinates plot
        ax1 = plt.subplot(2, 2, 1)
        for i in range(len(compositions)):
            color = plt.cm.viridis(predictions[i] / (predictions.max() - predictions.min() + 1e-8))
            ax1.plot(range(5), compositions[i], color=color, alpha=0.6, linewidth=0.8)
        
        ax1.set_xticks(range(5))
        ax1.set_xticklabels(elements, rotation=45)
        ax1.set_ylabel('Composition Fraction')
        ax1.set_title('Parallel Coordinates')
        ax1.grid(True, alpha=0.3)
        
        # 2. Composition vs Prediction scatter
        ax2 = plt.subplot(2, 2, 2)
        # Use first element fraction as x-axis
        scatter = ax2.scatter(compositions[:, 0], predictions, c=predictions, 
                            cmap='viridis', s=30, alpha=0.7)
        ax2.set_xlabel(f'{elements[0]} Fraction')
        ax2.set_ylabel('Predicted Value')
        ax2.set_title(f'Prediction vs {elements[0]} Content')
        ax2.grid(True, alpha=0.3)
        
        # 3. Correlation heatmap
        ax3 = plt.subplot(2, 2, 3)
        # Create correlation matrix between compositions and predictions
        corr_data = np.column_stack([compositions, predictions[:, np.newaxis]])
        corr_labels = elements + ['Prediction']
        corr_matrix = np.corrcoef(corr_data.T)
        
        im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(corr_labels)))
        ax3.set_yticks(range(len(corr_labels)))
        ax3.set_xticklabels(corr_labels, rotation=45)
        ax3.set_yticklabels(corr_labels)
        ax3.set_title('Correlation Matrix')
        
        # Add correlation values as text
        for i in range(len(corr_labels)):
            for j in range(len(corr_labels)):
                text = ax3.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        # 4. Distribution of predictions
        ax4 = plt.subplot(2, 2, 4)
        ax4.hist(predictions, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Predicted Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Predictions')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar for scatter plot
        cbar = fig.colorbar(scatter, ax=[ax1, ax2], shrink=0.8, aspect=30)
        cbar.set_label('Predicted Value', rotation=270, labelpad=20)
        
        plt.suptitle(f'Pentanary System: {element_key}', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(self.plots_dir, "pentanary", f"{element_key}_pentanary.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved pentanary diagram: {filename}")

    def create_phase_diagrams(self):
        """Create phase diagrams for all element combinations"""
        if self.results_df is None:
            print("No predictions available. Run predict() first.")
            return
        
        print("Creating phase diagrams...")
        element_groups = self.group_predictions_by_elements()
        
        for element_key, element_group in element_groups.items():
            num_elements = len(element_group['elements'])
            
            print(f"Creating diagram for {element_key} ({num_elements} elements)...")
            
            if num_elements == 3:
                self.plot_ternary_diagram(element_group, element_key)
            elif num_elements == 4:
                self.plot_quaternary_diagram(element_group, element_key)
            elif num_elements == 5:
                self.plot_pentanary_diagram(element_group, element_key)
        
        print(f"Phase diagrams saved in: {self.plots_dir}")

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
                    'System': values["System_Name"],
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
        Performs prediction, saves results to CSV, and creates phase diagrams.
        Returns the DataFrame containing predictions and targets.
        """
        print(f"Running prediction using model from {self.model_path} on data from {self.evaluation_path}")
        
        self.predict()

        if self.results_df is not None and not self.results_df.empty:
            # Save predictions to CSV
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
            
            # Create phase diagrams
            self.create_phase_diagrams()
            
        elif self.results_df is not None and self.results_df.empty:
             print("⚠️ Prediction ran, but the resulting DataFrame is empty. Check data or prediction process.")
        else:
            print("⚠️ Prediction did not generate a DataFrame. Results not saved.")
            
        return self.results_df

# Example Usage
if __name__ == '__main__':
    checkpoint_dir = 'checkpoints/AlloyTransformer_Regression_22_07__v3.3.1'
    config_path = os.path.join(checkpoint_dir, 'config.json')
    model_path = os.path.join(checkpoint_dir, 'final_model.pt')

    try:
        with open(config_path, 'r') as config_file:
            configs = json.load(config_file)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        configs = {}  # Provide default config or handle appropriately

    from glob import glob

    paths = glob("utils/quaternary/*csv")

    for path in paths:
        pred = PredictorWithPlots(
            model_path=model_path,
            evaluation_path=path,
            configs=configs,
            save_dir="predictions_with_plots"
        )
        
        # Run predictions and create plots
        results = pred()
        
        if results is not None:
            print(f"Prediction completed. Results shape: {results.shape}")
            print(f"Phase diagrams created in: {pred.plots_dir}")
        else:
            print("Prediction failed or returned no results.")