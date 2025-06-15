import os
import json
import datetime
from pathlib import Path
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
import pandas as pd
import os
import re
from collections import defaultdict
from glob import glob
from dataloader import LM_Dataset, collate_fn
from trainer import AlloyTransformerLightning  # Import the Lightning module


def parse_composition(composition_str):
    """
    Parse composition string to extract elements and their fractions
    Example: "Al0.19Mg0.81" -> {"Al": 0.19, "Mg": 0.81}
    """
    pattern = r'([A-Z][a-z]*)([0-9]*\.?[0-9]*)'
    matches = re.findall(pattern, composition_str)
    
    composition_dict = {}
    for element, fraction in matches:
        if fraction == "":
            fraction = "1"
        frac = float(fraction)
        if frac > 1:
            frac = frac / 100.0
        composition_dict[element] = frac
    
    # Normalize fractions to ensure they sum to 1
    total = sum(composition_dict.values())
    if total != 1.0:
        for element in composition_dict:
            composition_dict[element] /= total
    
    return composition_dict


def get_primary_element(composition_dict):
    """
    Get the element with highest fraction in the composition
    """
    if not composition_dict:
        return "Unknown"
    return max(composition_dict.items(), key=lambda x: x[1])[0]


def load_lightning_model(checkpoint_path, config=None):
    """
    Load a PyTorch Lightning model from checkpoint with improved handling of different architectures
    """
    # Add safe globals to handle PyTorch 2.6 security changes
    import torch.torch_version
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    
    if checkpoint_path.endswith('.ckpt'):
        # Load from Lightning checkpoint
        try:
            model = AlloyTransformerLightning.load_from_checkpoint(checkpoint_path, weights_only=False)
            print(f"Successfully loaded Lightning checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading Lightning checkpoint: {str(e)}")
            # Try with safe globals
            model = AlloyTransformerLightning.load_from_checkpoint(checkpoint_path)
    else:
        # Load from state dict with explicit weights_only=False
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            print(f"Successfully loaded checkpoint with weights_only=False")
        except Exception as e:
            print(f"Error loading with weights_only=False: {str(e)}")
            # Fallback to default with safe globals already added
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"Successfully loaded checkpoint after adding safe globals")
        
        print(f"Creating model with config: hidden_dim={config['d_model']}, " 
              f"num_layers={config['num_transformer_layers']}, num_heads={config['num_head']}")
        
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
                        model_inner_state_dict = {k.replace('model.', ''): v for k, v in model_state_dict.items() 
                                                 if k.startswith('model.')}
                        model.model.load_state_dict(model_inner_state_dict)
                        print("Successfully loaded inner model state dict.")
                        
                        # Load remaining parameters
                        non_model_state = {k: v for k, v in model_state_dict.items() if not k.startswith('model.')}
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


class CompositionEvaluator:
    def __init__(self, model_path, test_path, config=None, save_dir="eval"):
        self.model_path = model_path
        self.test_path = test_path
        self.config = config
        self.save_dir = save_dir
        
        # Create evaluation directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Load model
        self.model = load_lightning_model(model_path, config)
        
        # The model should already be setup from the load function
        # But call it again if needed
        if hasattr(self.model, 'model') and self.model.model is None:
            self.model.setup('test')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Load test data
        self.test_dataset = LM_Dataset(test_path)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn
        )
                                
        # Load compositions
        self.compositions = []
        with open(test_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                composition, _ = line.strip().split(',')
                self.compositions.append(composition)
        
        # Initialize results storage
        self.predictions = None
        self.targets = None
        self.results_df = None
        self.summary_stats = None
    
    def parse_composition(self, composition: str) -> list[tuple[str, float]]:
        """
        Parse composition string to extract elements and their fractions
        Example: "Al0.19Mg0.81" -> [("Al", 0.19), ("Mg", 0.81)]
        
        Args:
            composition: String representation of the composition (e.g., "Al0.19Mg0.81")
            
        Returns:
            List of tuples with elements and their normalized fractions
        """
        pattern = r'([A-Z][a-z]*)([0-9]*\.?[0-9]*)'
        matches = re.findall(pattern, composition)

        elements_fractions = []
        for element, fraction in matches:
            # Handle empty fraction (e.g., "Fe" instead of "Fe1.0")
            if fraction == "":
                fraction = "1"
            frac = float(fraction)
            if frac > 1:
                frac = frac / 100.0
            elements_fractions.append((element, frac))
        
        # Normalize fractions to ensure they sum to 1
        total = sum(frac for _, frac in elements_fractions)
        if total != 1.0:
            elements_fractions = [(element, frac/total) for element, frac in elements_fractions]
        
        # Pad with empty elements if needed (for fixed-length representation)
        while len(elements_fractions) < 4:
            elements_fractions.append(("", 0.0))
        return elements_fractions
    
    def predict(self):
        """Run model predictions on test set"""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).squeeze(-1)
                print(f"Starting prediction {inputs.shape}")
                predictions = self.model(inputs)
                
                all_predictions.extend(predictions.squeeze().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        self.predictions = np.array(all_predictions)
        self.targets = np.array(all_targets)
        
        return self.predictions, self.targets
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        if self.predictions is None or self.targets is None:
            self.predict()
        
        # Basic metrics
        mae = mean_absolute_error(self.targets, self.predictions)
        mse = mean_squared_error(self.targets, self.predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.targets, self.predictions)
        
        # Percentage errors
        percentage_errors = np.abs((self.targets - self.predictions) / self.targets) * 100
        mean_percentage_error = np.mean(percentage_errors)
        
        residuals = self.targets - self.predictions
        
        self.summary_stats = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Mean_Percentage_Error': mean_percentage_error,
            'Error_Std': np.std(residuals),
            'Error_25th_percentile': np.percentile(np.abs(residuals), 25),
            'Error_50th_percentile': np.percentile(np.abs(residuals), 50),
            'Error_75th_percentile': np.percentile(np.abs(residuals), 75),
            'Error_90th_percentile': np.percentile(np.abs(residuals), 90),
            'Error_95th_percentile': np.percentile(np.abs(residuals), 95),
            'Total_Samples': len(self.targets)
        }
        
        print("\nDetailed Test Results:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
        
        return self.summary_stats
    
    def create_results_dataframe(self):
        """Create comprehensive results dataframe with composition analysis"""
        if self.predictions is None or self.targets is None:
            self.predict()
        
        # Parse compositions
        parsed_compositions = [parse_composition(comp) for comp in self.compositions]
        primary_elements = [get_primary_element(comp) for comp in parsed_compositions]
        
        # Get all unique elements
        all_elements = set()
        for comp in parsed_compositions:
            all_elements.update(comp.keys())
        
        # Calculate errors
        residuals = self.targets - self.predictions
        percentage_errors = np.abs((self.targets - self.predictions) / self.targets) * 100
        
        # Create dataframe
        self.results_df = pd.DataFrame({
            'Composition': self.compositions,
            'Primary_Element': primary_elements,
            'Actual': self.targets,
            'Predicted': self.predictions,
            'Error': residuals,
            'Absolute_Error': np.abs(residuals),
            'Percentage_Error': percentage_errors
        })
        
        # Add element fractions
        for element in sorted(all_elements):
            element_fractions = []
            for comp in parsed_compositions:
                element_fractions.append(comp.get(element, 0.0))
            self.results_df[f'{element}_fraction'] = element_fractions
        
        # Add number of elements
        self.results_df['n_elements'] = [len([v for v in comp.values() if v > 0]) 
                                        for comp in parsed_compositions]
        
        return self.results_df
    
    def plot_basic_evaluation(self):
        """Create basic evaluation plots"""
        if self.results_df is None:
            self.create_results_dataframe()
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Scatter plot: Predicted vs Actual
        plt.subplot(3, 2, 1)
        plt.scatter(self.targets, self.predictions, alpha=0.6, edgecolors='none', s=30)
        plt.plot([self.targets.min(), self.targets.max()], 
                 [self.targets.min(), self.targets.max()], 
                 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Liquidus Temperature (K)', fontsize=12)
        plt.ylabel('Predicted Liquidus Temperature (K)', fontsize=12)
        plt.title(f'Predicted vs Actual Values\nR² = {self.summary_stats["R2"]:.4f}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Residual plot
        plt.subplot(3, 2, 2)
        residuals = self.results_df['Error']
        plt.scatter(self.predictions, residuals, alpha=0.6, edgecolors='none', s=30)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Values', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.title('Residual Plot', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 3. Error distribution
        plt.subplot(3, 2, 3)
        plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Residuals', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Residual Distribution\nMean: {np.mean(residuals):.2f}, Std: {np.std(residuals):.2f}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 4. Percentage error distribution
        plt.subplot(3, 2, 4)
        plt.hist(self.results_df['Percentage_Error'], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Percentage Error (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Percentage Error Distribution\nMean: {self.summary_stats["Mean_Percentage_Error"]:.2f}%', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 5. Q-Q plot
        from scipy import stats
        plt.subplot(3, 2, 5)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 6. Error vs Actual Temperature
        plt.subplot(3, 2, 6)
        plt.scatter(self.targets, self.results_df['Absolute_Error'], alpha=0.6, edgecolors='none', s=30)
        plt.xlabel('Actual Temperature (K)', fontsize=12)
        plt.ylabel('Absolute Error', fontsize=12)
        plt.title('Absolute Error vs Actual Temperature', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'test_evaluation_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_composition_analysis(self):
        """Create composition-specific error analysis plots"""
        if self.results_df is None:
            self.create_results_dataframe()
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        
        # 1. Error by primary element
        ax1 = axes[0, 0]
        element_errors = self.results_df.groupby('Primary_Element')['Absolute_Error'].agg(['mean', 'std', 'count'])
        element_errors = element_errors.sort_values('mean', ascending=False)
        
        ax1.bar(element_errors.index, element_errors['mean'], yerr=element_errors['std'], capsize=5)
        ax1.set_xlabel('Primary Element')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title('Error by Primary Element')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add sample count annotations
        for i, (elem, row) in enumerate(element_errors.iterrows()):
            ax1.text(i, row['mean'] + row['std'] + 5, f'n={int(row["count"])}', 
                    ha='center', va='bottom', fontsize=10)
        
        # 2. Box plot of errors by primary element
        ax2 = axes[0, 1]
        top_elements = element_errors.index[:10]  # Top 10 most common elements
        data_for_boxplot = []
        labels_for_boxplot = []
        
        for elem in top_elements:
            elem_data = self.results_df[self.results_df['Primary_Element'] == elem]['Absolute_Error']
            if len(elem_data) > 0:
                data_for_boxplot.append(elem_data.values)
                labels_for_boxplot.append(f'{elem} (n={len(elem_data)})')
        
        ax2.boxplot(data_for_boxplot, labels=labels_for_boxplot)
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Error Distribution by Primary Element')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3-6. Error vs element fraction (for top 4 elements)
        top_4_elements = list(element_errors.index[:4])
        for idx, element in enumerate(top_4_elements):
            ax = axes[1 + idx//2, idx%2]
            if f'{element}_fraction' in self.results_df.columns:
                fractions = self.results_df[f'{element}_fraction']
                errors = self.results_df['Absolute_Error']
                
                # Create scatter plot
                scatter = ax.scatter(fractions, errors, alpha=0.6, c=self.results_df['Actual'], cmap='viridis')
                ax.set_xlabel(f'{element} Fraction')
                ax.set_ylabel('Absolute Error')
                ax.set_title(f'Error vs {element} Content')
                
                # Add trend line if there are enough points
                if len(fractions[fractions > 0]) > 10:
                    mask = fractions > 0
                    z = np.polyfit(fractions[mask], errors[mask], 1)
                    p = np.poly1d(z)
                    ax.plot(fractions[mask], p(fractions[mask]), "r--", alpha=0.8, linewidth=2)
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax, label='Actual Temperature (K)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'composition_error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_element_count_impact(self):
        """
        Analyze how the number of elements in an alloy affects prediction accuracy
        """
        if self.results_df is None:
            self.create_results_dataframe()
        
        # Group by number of elements
        element_count_analysis = self.results_df.groupby('n_elements').agg({
            'Absolute_Error': ['mean', 'median', 'std', 'min', 'max', 'count'],
            'Percentage_Error': ['mean', 'median']
        })
        
        # Save to CSV
        element_count_analysis.to_csv(os.path.join(self.save_dir, 'element_count_analysis.csv'))
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Mean error by element count
        ax1 = plt.subplot(2, 1, 1)
        bars = ax1.bar(element_count_analysis.index, element_count_analysis[('Absolute_Error', 'mean')])
        
        # Add count labels
        for bar, count in zip(bars, element_count_analysis[('Absolute_Error', 'count')]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'n={int(count)}', ha='center', va='bottom', rotation=0)
        
        ax1.set_xlabel('Number of Elements in Composition')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title('Mean Prediction Error by Number of Elements')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot of errors by element count
        ax2 = plt.subplot(2, 1, 2)
        
        element_counts = sorted(self.results_df['n_elements'].unique())
        data_for_boxplot = [self.results_df[self.results_df['n_elements'] == count]['Absolute_Error'] 
                            for count in element_counts]
        
        ax2.boxplot(data_for_boxplot, labels=[f'{count} elements' for count in element_counts])
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Error Distribution by Number of Elements')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'element_count_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nImpact of Element Count on Prediction Error:")
        print(element_count_analysis)
        
        return element_count_analysis

    def analyze_entropy_effect(self):
        """
        Analyze how compositional entropy affects prediction accuracy
        Entropy is higher when elements are more evenly distributed
        """
        if self.results_df is None:
            self.create_results_dataframe()
        
        # Calculate compositional entropy for each alloy
        entropies = []
        
        for idx, row in self.results_df.iterrows():
            composition = row['Composition']
            elements_fractions = self.parse_composition(composition)
            
            # Calculate Shannon entropy: -sum(p * log(p))
            entropy = 0
            for _, frac in elements_fractions:
                if frac > 0:
                    entropy -= frac * np.log(frac)
            
            entropies.append(entropy)
        
        # Add entropy to results dataframe
        self.results_df['Compositional_Entropy'] = entropies
        
        # Create entropy bands for analysis
        self.results_df['Entropy_Band'] = pd.cut(
            self.results_df['Compositional_Entropy'], 
            bins=5, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Analyze error by entropy band
        entropy_analysis = self.results_df.groupby('Entropy_Band').agg({
            'Absolute_Error': ['mean', 'median', 'std', 'count'],
            'Percentage_Error': ['mean', 'median']
        })
        
        # Save to CSV
        entropy_analysis.to_csv(os.path.join(self.save_dir, 'entropy_analysis.csv'))
        
        # Visualize relationship between entropy and error
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Scatter plot of entropy vs error
        ax1 = plt.subplot(2, 2, 1)
        scatter = ax1.scatter(
            self.results_df['Compositional_Entropy'], 
            self.results_df['Absolute_Error'],
            alpha=0.6, 
            c=self.results_df['n_elements'],
            cmap='viridis',
            s=50
        )
        
        # Add trend line
        z = np.polyfit(self.results_df['Compositional_Entropy'], self.results_df['Absolute_Error'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(self.results_df['Compositional_Entropy'].min(), 
                            self.results_df['Compositional_Entropy'].max(), 100)
        ax1.plot(x_range, p(x_range), "r--", linewidth=2)
        
        # Add correlation coefficient
        corr = np.corrcoef(self.results_df['Compositional_Entropy'], self.results_df['Absolute_Error'])[0, 1]
        ax1.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax1.transAxes, 
                fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax1.set_xlabel('Compositional Entropy', fontsize=12)
        ax1.set_ylabel('Absolute Error', fontsize=12)
        ax1.set_title('Prediction Error vs Compositional Entropy', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar for number of elements
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Number of Elements', fontsize=12)
        
        # Plot 2: Bar chart of mean error by entropy band
        ax2 = plt.subplot(2, 2, 2)
        bars = ax2.bar(entropy_analysis.index, entropy_analysis[('Absolute_Error', 'mean')])
        
        # Add count labels
        for bar, count in zip(bars, entropy_analysis[('Absolute_Error', 'count')]):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'n={int(count)}', ha='center', va='bottom', rotation=0)
        
        ax2.set_xlabel('Entropy Band', fontsize=12)
        ax2.set_ylabel('Mean Absolute Error', fontsize=12)
        ax2.set_title('Mean Error by Compositional Entropy', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Box plot of errors by entropy band
        ax3 = plt.subplot(2, 2, 3)
        sns.boxplot(x='Entropy_Band', y='Absolute_Error', data=self.results_df, ax=ax3)
        ax3.set_xlabel('Entropy Band', fontsize=12)
        ax3.set_ylabel('Absolute Error', fontsize=12)
        ax3.set_title('Error Distribution by Compositional Entropy', fontsize=14)
        
        # Plot 4: Entropy vs percentage error
        ax4 = plt.subplot(2, 2, 4)
        scatter = ax4.scatter(
            self.results_df['Compositional_Entropy'], 
            self.results_df['Percentage_Error'],
            alpha=0.6, 
            c=self.results_df['Actual'],
            cmap='plasma',
            s=50
        )
        
        z = np.polyfit(self.results_df['Compositional_Entropy'], self.results_df['Percentage_Error'], 1)
        p = np.poly1d(z)
        ax4.plot(x_range, p(x_range), "r--", linewidth=2)
        
        corr = np.corrcoef(self.results_df['Compositional_Entropy'], self.results_df['Percentage_Error'])[0, 1]
        ax4.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax4.transAxes, 
                fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax4.set_xlabel('Compositional Entropy', fontsize=12)
        ax4.set_ylabel('Percentage Error (%)', fontsize=12)
        ax4.set_title('Percentage Error vs Compositional Entropy', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Actual Temperature (K)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'entropy_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nImpact of Compositional Entropy on Prediction Error:")
        print(entropy_analysis)
        
        return entropy_analysis

    def analyze_elemental_similarity(self):
        """
        Analyze if compositions with similar elements but different ratios 
        have consistent prediction errors
        """
        if self.results_df is None:
            self.create_results_dataframe()
        
        # Find all unique elements in the dataset
        element_columns = [col for col in self.results_df.columns if col.endswith('_fraction')]
        elements = [col.split('_')[0] for col in element_columns]
        elements = [e for e in elements if e]  # Remove empty strings
        
        # Get element sets (combinations of elements used in compositions)
        element_sets = []
        element_set_to_idx = {}
        
        for idx, row in self.results_df.iterrows():
            # Get elements present in this composition
            present_elements = set()
            for element in elements:
                if row[f'{element}_fraction'] > 0:
                    present_elements.add(element)
            
            present_elements = tuple(sorted(present_elements))
            
            if present_elements:
                if present_elements not in element_set_to_idx:
                    element_set_to_idx[present_elements] = len(element_sets)
                    element_sets.append(present_elements)
        
        # Add element set index to dataframe
        self.results_df['Element_Set'] = -1
        for idx, row in self.results_df.iterrows():
            present_elements = tuple(sorted([
                element for element in elements 
                if row[f'{element}_fraction'] > 0
            ]))
            
            if present_elements in element_set_to_idx:
                self.results_df.at[idx, 'Element_Set'] = element_set_to_idx[present_elements]
        
        # Find element sets with at least 3 compositions for meaningful analysis
        element_set_counts = self.results_df['Element_Set'].value_counts()
        common_element_sets = element_set_counts[element_set_counts >= 3].index.tolist()
        
        if not common_element_sets:
            print("No element sets with at least 3 compositions found. Skipping elemental similarity analysis.")
            return None
        
        # Create consolidated dataframe for analysis
        element_set_data = []
        
        for set_idx in common_element_sets:
            set_compositions = self.results_df[self.results_df['Element_Set'] == set_idx]
            element_names = element_sets[set_idx]
            
            element_set_data.append({
                'Element_Set': '-'.join(element_names),
                'Count': len(set_compositions),
                'Mean_Error': set_compositions['Absolute_Error'].mean(),
                'Std_Error': set_compositions['Absolute_Error'].std(),
                'Mean_Pct_Error': set_compositions['Percentage_Error'].mean(),
                'Min_Temp': set_compositions['Actual'].min(),
                'Max_Temp': set_compositions['Actual'].max(),
                'Temp_Range': set_compositions['Actual'].max() - set_compositions['Actual'].min()
            })
        
        element_set_df = pd.DataFrame(element_set_data)
        element_set_df = element_set_df.sort_values('Count', ascending=False)
        
        # Save to CSV
        element_set_df.to_csv(os.path.join(self.save_dir, 'element_set_analysis.csv'), index=False)
        
        print("\nElement Set Analysis (Top 10):")
        print(element_set_df.head(10))
        
        return element_set_df

    def save_results(self):
        """Save all evaluation results"""
        if self.results_df is None:
            self.create_results_dataframe()
        
        # Save detailed results
        self.results_df.to_csv(os.path.join(self.save_dir, 'test_predictions_detailed.csv'), index=False)
        
        # Save error by primary element
        element_errors = self.results_df.groupby('Primary_Element')['Absolute_Error'].agg(['mean', 'std', 'count'])
        element_errors.to_csv(os.path.join(self.save_dir, 'error_by_primary_element.csv'))
        
        # Save worst predictions
        worst_predictions = self.results_df.nlargest(20, 'Absolute_Error')
        worst_predictions.to_csv(os.path.join(self.save_dir, 'worst_predictions.csv'), index=False)
        
        # Save summary statistics
        with open(os.path.join(self.save_dir, 'evaluation_summary.txt'), 'w') as f:
            f.write("Test Set Evaluation Summary\n")
            f.write("===========================\n\n")
            for key, value in self.summary_stats.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"\nResults saved to {self.save_dir}/ folder")
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline with enhanced element analysis"""
        print("Starting model evaluation...")
        
        # Get predictions
        self.predict()
        
        print("Calculating metrics...")
        # Calculate metrics
        self.calculate_metrics()
        
        # Create results dataframe
        self.create_results_dataframe()
        
        # Basic plots
        print("Creating basic evaluation plots...")
        self.plot_basic_evaluation()
        self.plot_composition_analysis()
        
        # Enhanced element analysis
        print("Performing enhanced element analysis...")
        
        # Analysis of element count impact
        print("Analyzing impact of element count...")
        self.analyze_element_count_impact()
        
        # Analysis of compositional entropy
        print("Analyzing effect of compositional entropy...")
        self.analyze_entropy_effect()
        
        # Analysis of elemental similarity
        print("Analyzing elemental similarity patterns...")
        self.analyze_elemental_similarity()
        
        # Save results
        print("Saving results...")
        self.save_results()
        
        return self.results_df, self.summary_stats


class DiagramEvaluator:
    def __init__(self, paths: str, model):
        self.paths = paths
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load diagrams from CSV files
        self.diagrams = []
        csv_files = glob(os.path.join(paths, "**/*.csv"), recursive=True)
        
        if not csv_files:
            print(f"Warning: No CSV files found in {paths}")
            return
            
        print(f"Found {len(csv_files)} CSV files in {paths}")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Add filename as metadata
                df['source_file'] = os.path.basename(csv_file)
                self.diagrams.append(df)
                print(f"Loaded diagram from {os.path.basename(csv_file)} with {len(df)} points")
            except Exception as e:
                print(f"Error loading {csv_file}: {str(e)}")
        
        if not self.diagrams:
            print("No valid diagrams loaded!")
            return
            
        # Combine all diagrams
        self.combined_diagram = pd.concat(self.diagrams, ignore_index=True)
        print(f"Combined diagram has {len(self.combined_diagram)} total points")
        
        # Initialize results storage
        self.predictions = {}
        self.targets = {}
        self.compositions = {}
        
    def prepare_data_for_prediction(self, df):
        """
        Prepare diagram data for model prediction
        Assumes the CSV has columns for composition and temperature
        """
        # Try to identify composition and temperature columns
        composition_col = None
        temperature_col = None
        
        # Common column names for composition
        comp_names = ['composition', 'Composition', 'comp', 'alloy', 'formula']
        for col in comp_names:
            if col in df.columns:
                composition_col = col
                break
        
        # Common column names for temperature
        temp_names = ['temperature', 'Temperature', 'temp', 'T', 'liquidus', 'Liquidus']
        for col in temp_names:
            if col in df.columns:
                temperature_col = col
                break
        
        if composition_col is None or temperature_col is None:
            print(f"Warning: Could not identify composition and temperature columns")
            print(f"Available columns: {list(df.columns)}")
            # Try to use first two columns as fallback
            if len(df.columns) >= 2:
                composition_col = df.columns[0]
                temperature_col = df.columns[1]
                print(f"Using fallback: {composition_col} for composition, {temperature_col} for temperature")
            else:
                return None, None
        
        compositions = df[composition_col].tolist()
        temperatures = df[temperature_col].tolist()
        
        return compositions, temperatures
    
    def create_dataset_from_compositions(self, compositions):
        """
        Create a dataset from compositions that can be fed to the model
        This mimics the LM_Dataset structure
        """
        # Create a temporary CSV file with the compositions
        temp_data = []
        for comp in compositions:
            # Use dummy temperature since we're only predicting
            temp_data.append([comp, 1000.0])  # dummy temperature
        
        temp_df = pd.DataFrame(temp_data, columns=['composition', 'temperature'])
        temp_file = os.path.join(self.paths, 'temp_diagram_data.csv')
        temp_df.to_csv(temp_file, index=False)
        
        try:
            # Create dataset
            dataset = LM_Dataset(temp_file)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
            return dataloader
        except Exception as e:
            print(f"Error creating dataset: {str(e)}")
            return None
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def predict_diagram(self, diagram_name=None):
        """
        Run model predictions on phase diagram data
        """
        if not self.diagrams:
            print("No diagrams loaded!")
            return
        
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        
        # If specific diagram name is provided, predict only that one
        if diagram_name:
            target_diagrams = [df for df in self.diagrams if df['source_file'].iloc[0] == diagram_name]
            if not target_diagrams:
                print(f"Diagram {diagram_name} not found!")
                return
            diagrams_to_predict = target_diagrams
        else:
            diagrams_to_predict = self.diagrams
        
        for df in diagrams_to_predict:
            source_file = df['source_file'].iloc[0]
            print(f"Predicting diagram: {source_file}")
            
            # Prepare data
            compositions, actual_temperatures = self.prepare_data_for_prediction(df)
            if compositions is None:
                continue
            
            # Create dataloader
            dataloader = self.create_dataset_from_compositions(compositions)
            if dataloader is None:
                continue
            
            # Make predictions
            predictions = []
            with torch.no_grad():
                for inputs, _ in dataloader:  # We ignore the dummy targets
                    inputs = inputs.to(self.device)
                    pred = self.model(inputs)
                    
                    # Handle different prediction shapes
                    pred_numpy = pred.squeeze().cpu().numpy()
                    
                    # Ensure we have an iterable array
                    if pred_numpy.ndim == 0:  # scalar
                        predictions.append(float(pred_numpy))
                    elif pred_numpy.ndim == 1:  # 1D array
                        predictions.extend(pred_numpy.tolist())
                    else:  # higher dimensional
                        predictions.extend(pred_numpy.flatten().tolist())
            
            # Store results
            self.predictions[source_file] = np.array(predictions)
            self.targets[source_file] = np.array(actual_temperatures)
            self.compositions[source_file] = compositions
            
            print(f"Completed predictions for {source_file}: {len(predictions)} points")
    
    def calculate_diagram_metrics(self, diagram_name=None):
        """
        Calculate metrics for phase diagram predictions
        """
        if diagram_name and diagram_name in self.predictions:
            diagrams_to_analyze = [diagram_name]
        else:
            diagrams_to_analyze = list(self.predictions.keys())
        
        metrics_summary = {}
        
        for name in diagrams_to_analyze:
            if name not in self.predictions:
                continue
                
            pred = self.predictions[name]
            target = self.targets[name]
            
            # Convert to numpy arrays if they aren't already
            pred = np.array(pred)
            target = np.array(target)
            
            # Ensure same length
            min_len = min(len(pred), len(target))
            pred = pred[:min_len]
            target = target[:min_len]
            
            # Skip if we don't have enough data points
            if min_len < 2:
                print(f"Warning: Not enough data points for {name} (only {min_len} points)")
                continue
            
            # Calculate metrics with error handling
            try:
                mae = mean_absolute_error(target, pred)
                rmse = np.sqrt(mean_squared_error(target, pred))
                r2 = r2_score(target, pred)
                
                # Percentage errors (avoid division by zero)
                with np.errstate(divide='ignore', invalid='ignore'):
                    percentage_errors = np.abs((target - pred) / target) * 100
                    # Remove infinite and NaN values
                    percentage_errors = percentage_errors[np.isfinite(percentage_errors)]
                    mean_percentage_error = np.mean(percentage_errors) if len(percentage_errors) > 0 else 0
                
                metrics_summary[name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'Mean_Percentage_Error': mean_percentage_error,
                    'Points': len(pred)
                }
                
                print(f"\nMetrics for {name}:")
                print(f"  MAE: {mae:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  R²: {r2:.4f}")
                print(f"  Mean Percentage Error: {mean_percentage_error:.2f}%")
                print(f"  Points: {len(pred)}")
                
            except Exception as e:
                print(f"Error calculating metrics for {name}: {str(e)}")
                continue
        
        return metrics_summary
    
    def plot_phase_diagram_comparison(self, diagram_name=None, save_dir="diagram_eval"):
        """
        Plot phase diagram comparisons between real and predicted values
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if diagram_name and diagram_name in self.predictions:
            diagrams_to_plot = [diagram_name]
        else:
            diagrams_to_plot = list(self.predictions.keys())
        
        for name in diagrams_to_plot:
            if name not in self.predictions:
                continue
                
            pred = np.array(self.predictions[name])
            target = np.array(self.targets[name])
            compositions = self.compositions[name]
            
            # Ensure same length
            min_len = min(len(pred), len(target), len(compositions))
            if min_len < 2:
                print(f"Warning: Not enough data points for {name} (only {min_len} points). Skipping plot.")
                continue
                
            pred = pred[:min_len]
            target = target[:min_len]
            compositions = compositions[:min_len]
            
            try:
                # Create figure with subplots
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'Phase Diagram Analysis: {name}', fontsize=16)
                
                # Plot 1: Predicted vs Actual scatter plot
                ax1 = axes[0, 0]
                ax1.scatter(target, pred, alpha=0.6, s=30)
                min_val = min(target.min(), pred.min())
                max_val = max(target.max(), pred.max())
                ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
                ax1.set_xlabel('Actual Temperature (K)')
                ax1.set_ylabel('Predicted Temperature (K)')
                ax1.set_title('Predicted vs Actual Values')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Phase diagram overlay
                ax2 = axes[0, 1]
                x_values = np.arange(len(target))  # Use index as x-axis for now
                ax2.plot(x_values, target, 'o-', linewidth=2, markersize=4, 
                        color='blue', alpha=0.7, label='Actual')
                ax2.plot(x_values, pred, 's-', linewidth=2, markersize=4, 
                        color='red', alpha=0.7, label='Predicted')
                ax2.set_xlabel('Composition Index')
                ax2.set_ylabel('Temperature (K)')
                ax2.set_title('Phase Diagram Comparison')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Error distribution
                ax3 = axes[1, 0]
                errors = target - pred
                ax3.hist(errors, bins=min(30, len(errors)//2), alpha=0.7, edgecolor='black')
                ax3.set_xlabel('Prediction Error (K)')
                ax3.set_ylabel('Frequency')
                ax3.set_title(f'Error Distribution\nMean: {np.mean(errors):.2f}K, Std: {np.std(errors):.2f}K')
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Error vs actual temperature
                ax4 = axes[1, 1]
                ax4.scatter(target, np.abs(errors), alpha=0.6, s=30)
                ax4.set_xlabel('Actual Temperature (K)')
                ax4.set_ylabel('Absolute Error (K)')
                ax4.set_title('Absolute Error vs Actual Temperature')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                safe_name = name.replace('.csv', '').replace(' ', '_').replace('/', '_')
                plt.savefig(os.path.join(save_dir, f'phase_diagram_{safe_name}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved phase diagram analysis for {name}")
                
            except Exception as e:
                print(f"Error creating plot for {name}: {str(e)}")
                plt.close()
                continue
    
    def save_diagram_results(self, save_dir="diagram_eval"):
        """
        Save phase diagram evaluation results
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save detailed predictions for each diagram
        for name in self.predictions.keys():
            if name not in self.predictions:
                continue
                
            pred = np.array(self.predictions[name])
            target = np.array(self.targets[name])
            compositions = self.compositions[name]
            
            # Create results dataframe
            min_len = min(len(pred), len(target), len(compositions))
            if min_len == 0:
                print(f"Warning: No data for {name}. Skipping.")
                continue
                
            try:
                results_df = pd.DataFrame({
                    'Composition': compositions[:min_len],
                    'Actual_Temperature': target[:min_len],
                    'Predicted_Temperature': pred[:min_len],
                    'Error': target[:min_len] - pred[:min_len],
                    'Absolute_Error': np.abs(target[:min_len] - pred[:min_len])
                })
                
                # Add percentage error with safe division
                with np.errstate(divide='ignore', invalid='ignore'):
                    percentage_errors = np.abs((target[:min_len] - pred[:min_len]) / target[:min_len]) * 100
                    percentage_errors = np.where(np.isfinite(percentage_errors), percentage_errors, 0)
                    results_df['Percentage_Error'] = percentage_errors
                
                # Save to CSV
                safe_name = name.replace('.csv', '').replace(' ', '_').replace('/', '_')
                results_df.to_csv(os.path.join(save_dir, f'diagram_results_{safe_name}.csv'), index=False)
                print(f"Saved results for {name}")
                
            except Exception as e:
                print(f"Error saving results for {name}: {str(e)}")
                continue
        
        # Save summary metrics
        try:
            metrics = self.calculate_diagram_metrics()
            if metrics:
                metrics_df = pd.DataFrame(metrics).T
                metrics_df.to_csv(os.path.join(save_dir, 'diagram_metrics_summary.csv'))
                print("Saved metrics summary")
        except Exception as e:
            print(f"Error saving metrics summary: {str(e)}")
        
        print(f"Diagram evaluation results saved to {save_dir}/")
    
    def run_full_diagram_evaluation(self, save_dir="diagram_eval"):
        """
        Run complete phase diagram evaluation pipeline
        """
        print("Starting phase diagram evaluation...")
        
        if not self.diagrams:
            print("No diagrams to evaluate!")
            return
        
        # Make predictions
        print("Generating predictions...")
        self.predict_diagram()
        
        if not self.predictions:
            print("No predictions generated!")
            return
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = self.calculate_diagram_metrics()
        
        # Create plots
        print("Creating phase diagram plots...")
        self.plot_phase_diagram_comparison(save_dir=save_dir)
        
        # Save results
        print("Saving results...")
        self.save_diagram_results(save_dir=save_dir)
        
        print(f"Phase diagram evaluation completed. Results saved to {save_dir}/")
        return metrics


def main():
    # Model checkpoint directory
    checkpoint_dir = 'checkpoints/AlloyTransformer_Regression_15_06__v0.1.1'
    config_path = os.path.join(checkpoint_dir, 'config.json')
    
    # Load configuration with proper error handling
    try:
        with open(config_path, 'r') as config_file:
            configs = json.load(config_file)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        return
    except json.JSONDecodeError:
        print(f"Error parsing JSON in {config_path}")
        return
    
    # Set model path
    model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    test_path = 'Data/All_Data_Based_06-06-2025/test_set_for_low_temp.csv'
    diagrams_paths = "Created Data"
    
    # Create results directories
    composition_results_dir = "alloy_transformer_results_6"
    diagram_results_dir = "diagram_evaluation_results"
    Path(composition_results_dir).mkdir(exist_ok=True)
    Path(diagram_results_dir).mkdir(exist_ok=True)
    
    # Load model once for both evaluators
    print("Loading model...")
    model = load_lightning_model(model_path, configs)
    
    # Run composition evaluation
    print("\n" + "="*50)
    print("COMPOSITION EVALUATION")
    print("="*50)
    
    evaluator = CompositionEvaluator(
        model_path=model_path,
        test_path=test_path,
        config=configs,
        save_dir=composition_results_dir
    )
    
    try:
        results_df, summary_stats = evaluator.run_full_evaluation()
        print(f"Composition evaluation completed successfully. Results saved to {composition_results_dir}")
    except Exception as e:
        print(f"Error during composition evaluation: {str(e)}")
    
    # # Run diagram evaluation
    # print("\n" + "="*50)
    # print("PHASE DIAGRAM EVALUATION")
    # print("="*50)
    
    # diagram_evaluator = DiagramEvaluator(paths=diagrams_paths, model=model)
    
    # try:
    #     diagram_metrics = diagram_evaluator.run_full_diagram_evaluation(save_dir=diagram_results_dir)
    #     print(f"Diagram evaluation completed successfully. Results saved to {diagram_results_dir}")
    # except Exception as e:
    #     print(f"Error during diagram evaluation: {str(e)}")


if __name__ == "__main__":
    main()