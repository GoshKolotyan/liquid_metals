import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
from glob import glob
import re
from datetime import datetime
from typing import List, Dict, Optional
import time

# Import your existing modules
from trainer import AlloyTransformerLightning 
from dataloader import LM_Dataset, collate_fn 

class CompletePredictionPipeline:
    def __init__(self, model_path: str, configs: dict, base_data_dir: str = "utils", save_dir: str = "complete_predictions"):
        self.model_path = model_path
        self.configs = configs
        self.base_data_dir = base_data_dir
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize model
        print("🔄 Loading model...")
        self.model = self.load_lightning_model(self.model_path, self.configs)
        self.model.to(self.device)
        self.model.eval()
        
        # Ensure model is properly set up
        if hasattr(self.model, 'model') and self.model.model is None:
            print("Running setup for the model...")
            self.model.setup('test')
        
        print(f"✅ Model loaded and ready on {self.device}")
        
        # Storage for all results
        self.all_results = []
        self.file_stats = []

    @staticmethod
    def load_lightning_model(checkpoint_path, config=None):
        """Load PyTorch Lightning model from checkpoint"""
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
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print(f"Successfully loaded checkpoint (older PyTorch version)")
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                raise e
        
            if config is None:
                raise ValueError("Config must be provided when loading from state dict")
                
            model = AlloyTransformerLightning(config)
            model.setup('test')
            
            # Extract and load state dict
            model_state_dict = None
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model_state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    model_state_dict = checkpoint['state_dict']
                else:
                    model_state_dict = checkpoint
            else:
                model_state_dict = checkpoint
            
            try:
                model.load_state_dict(model_state_dict)
                print("Successfully loaded model state dict directly.")
            except RuntimeError as e:
                print(f"Direct loading failed: {e}")
                if hasattr(model, 'model') and model.model is not None:
                    if any(key.startswith('model.') for key in model_state_dict.keys()):
                        model_inner_state_dict = {
                            k.replace('model.', ''): v for k, v in model_state_dict.items() 
                            if k.startswith('model.')
                        }
                        model.model.load_state_dict(model_inner_state_dict)
                        print("Successfully loaded inner model state dict.")
                    else:
                        model.model.load_state_dict(model_state_dict, strict=False)
                        print("Loaded state dict into inner model with strict=False.")
                else:
                    model.load_state_dict(model_state_dict, strict=False)
                    print("Loaded state dict with strict=False.")
        
        model.eval()
        return model

    def find_all_csv_files(self) -> List[Dict[str, str]]:
        """Find all CSV files in utils subdirectories"""
        file_info = []
        
        # Define the subdirectories to search
        subdirs = ['pentanary', 'quaternary', 'ternary']
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.base_data_dir, subdir)
            if not os.path.exists(subdir_path):
                print(f"⚠️  Directory not found: {subdir_path}")
                continue
                
            # Find all CSV files in this subdirectory
            csv_files = glob(os.path.join(subdir_path, "*.csv"))
            
            for csv_file in csv_files:
                file_info.append({
                    'full_path': csv_file,
                    'filename': os.path.basename(csv_file),
                    'alloy_type': subdir,
                    'file_size': os.path.getsize(csv_file)
                })
        
        return file_info

    def predict_single_file(self, file_path: str, alloy_type: str, filename: str) -> Optional[pd.DataFrame]:
        """Predict on a single CSV file"""
        try:
            # Create dataset and dataloader
            test_dataset = LM_Dataset(file_path, has_targets=False)
            test_loader = DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False,
                collate_fn=collate_fn
            )
            
            # Run predictions
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    # Handle different batch formats
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        inputs, targets = batch
                    else:
                        inputs = batch
                        targets = None
                    
                    inputs = inputs.to(self.device)
                    if targets is not None:
                        targets = targets.to(self.device)

                    # Get predictions
                    predictions = self.model(inputs)
                    
                    # Handle prediction shapes
                    predictions_np = predictions.cpu().numpy()
                    if predictions_np.ndim > 1:
                        if predictions_np.shape[-1] == 1:
                            predictions_np = predictions_np.squeeze(-1)
                        else:
                            predictions_np = predictions_np[:, 0]  # Take first output
                    
                    # Handle targets if they exist
                    if targets is not None:
                        targets_np = targets.cpu().numpy()
                        if targets_np.ndim > 1:
                            if targets_np.shape[-1] == 1:
                                targets_np = targets_np.squeeze(-1)
                            else:
                                targets_np = targets_np[:, 0]  # Take first target
                        
                        if targets_np.ndim > 1:
                            targets_np = targets_np.flatten()
                        
                        all_targets.extend(targets_np.tolist())
                    else:
                        all_targets.extend([None] * len(predictions_np))
                    
                    # Ensure predictions are 1D
                    if predictions_np.ndim > 1:
                        predictions_np = predictions_np.flatten()
                    
                    all_predictions.extend(predictions_np.tolist())
            
            if not all_predictions:
                print(f"❌ No predictions generated for {filename}")
                return None
            
            # Load original data to get system names
            original_data = pd.read_csv(file_path)
            
            # Handle length mismatch
            min_length = min(len(all_predictions), len(original_data))
            if len(all_predictions) != len(original_data):
                print(f"⚠️  Length mismatch in {filename}: predictions={len(all_predictions)}, data={len(original_data)}")
                all_predictions = all_predictions[:min_length]
                if all_targets[0] is not None:
                    all_targets = all_targets[:min_length]
                original_data = original_data.iloc[:min_length]
            
            # Create DataFrame for this file
            result_data = {
                'System_Name': original_data['System_Name'].values,
                'Alloy_Type': [alloy_type] * len(all_predictions),
                'Source_File': [filename] * len(all_predictions),
                'Prediction': all_predictions
            }
            
            # Add targets if available
            if all_targets[0] is not None:
                result_data['Target'] = all_targets
            
            # Add composition analysis
            result_data['Num_Components'] = [self.count_components(system) for system in result_data['System_Name']]
            
            result_df = pd.DataFrame(result_data)
            
            print(f"✅ {filename}: {len(result_df)} predictions generated")
            return result_df
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            return None

    def count_components(self, system_name: str) -> int:
        """Count number of components in a system name"""
        try:
            pattern = r'([A-Z][a-z]*)([0-9]*\.?[0-9]+)'
            matches = re.findall(pattern, system_name)
            return len(matches)
        except:
            return 0

    def run_complete_pipeline(self):
        """Run prediction pipeline on all files"""
        print("🚀 Starting Complete Alloy Prediction Pipeline")
        print("=" * 60)
        
        # Find all CSV files
        all_files = self.find_all_csv_files()
        
        if not all_files:
            print("❌ No CSV files found!")
            return None
        
        print(f"📋 Found {len(all_files)} CSV files to process:")
        for file_info in all_files:
            print(f"   📁 {file_info['alloy_type']}/{file_info['filename']} ({file_info['file_size']} bytes)")
        
        print("\n🔄 Starting predictions...")
        print("-" * 60)
        
        # Process each file
        successful_files = 0
        failed_files = 0
        start_time = time.time()
        
        for i, file_info in enumerate(all_files, 1):
            print(f"\n📊 Processing {i}/{len(all_files)}: {file_info['alloy_type']}/{file_info['filename']}")
            
            result_df = self.predict_single_file(
                file_info['full_path'],
                file_info['alloy_type'],
                file_info['filename']
            )
            
            if result_df is not None:
                # Add to overall results
                self.all_results.append(result_df)
                
                # Track statistics
                self.file_stats.append({
                    'filename': file_info['filename'],
                    'alloy_type': file_info['alloy_type'],
                    'num_predictions': len(result_df),
                    'min_prediction': result_df['Prediction'].min(),
                    'max_prediction': result_df['Prediction'].max(),
                    'mean_prediction': result_df['Prediction'].mean(),
                    'has_targets': 'Target' in result_df.columns,
                    'components_range': f"{result_df['Num_Components'].min()}-{result_df['Num_Components'].max()}"
                })
                
                successful_files += 1
            else:
                self.file_stats.append({
                    'filename': file_info['filename'],
                    'alloy_type': file_info['alloy_type'],
                    'status': 'FAILED'
                })
                failed_files += 1
        
        # Combine all results
        if self.all_results:
            print(f"\n🔗 Combining all results...")
            combined_df = pd.concat(self.all_results, ignore_index=True)
            
            # Add timestamp and processing info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_df['Processing_Timestamp'] = timestamp
            combined_df['Prediction_Index'] = range(len(combined_df))
            
            # Save combined results
            output_path = os.path.join(self.save_dir, f"all_alloy_predictions_{timestamp}.csv")
            combined_df.to_csv(output_path, index=False)
            
            # Also save a version without timestamp for easy access
            simple_output_path = os.path.join(self.save_dir, "all_alloy_predictions_latest.csv")
            combined_df.to_csv(simple_output_path, index=False)
            
            # Save statistics
            stats_df = pd.DataFrame(self.file_stats)
            stats_path = os.path.join(self.save_dir, f"prediction_statistics_{timestamp}.csv")
            stats_df.to_csv(stats_path, index=False)
            
            # Print summary
            end_time = time.time()
            processing_time = end_time - start_time
            
            print("\n" + "=" * 80)
            print("🎯 COMPLETE PIPELINE SUMMARY")
            print("=" * 80)
            print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
            print(f"📁 Files processed: {len(all_files)}")
            print(f"✅ Successful: {successful_files}")
            print(f"❌ Failed: {failed_files}")
            print(f"🔢 Total predictions: {len(combined_df)}")
            print(f"📊 Results saved to:")
            print(f"   📝 Combined results: {output_path}")
            print(f"   📝 Latest results: {simple_output_path}")
            print(f"   📊 Statistics: {stats_path}")
            
            # Print breakdown by alloy type
            print(f"\n📈 Breakdown by alloy type:")
            type_summary = combined_df.groupby('Alloy_Type').agg({
                'Prediction': ['count', 'mean', 'min', 'max'],
                'Num_Components': ['mean', 'min', 'max']
            }).round(3)
            print(type_summary)
            
            # Print prediction statistics
            print(f"\n📊 Overall prediction statistics:")
            print(f"   Min prediction: {combined_df['Prediction'].min():.3f}")
            print(f"   Max prediction: {combined_df['Prediction'].max():.3f}")
            print(f"   Mean prediction: {combined_df['Prediction'].mean():.3f}")
            print(f"   Std prediction: {combined_df['Prediction'].std():.3f}")
            
            return combined_df
        
        else:
            print("❌ No successful predictions generated!")
            return None

def main():
    """Main function to run the complete pipeline"""
    # Configuration
    checkpoint_dir = 'checkpoints/AlloyTransformer_Regression_27_07__v1.1.1'
    config_path = os.path.join(checkpoint_dir, 'config.json')
    model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    
    # Load config
    try:
        with open(config_path, 'r') as config_file:
            configs = json.load(config_file)
        print(f"✅ Config loaded from: {config_path}")
    except FileNotFoundError:
        print(f"❌ Config file not found at {config_path}")
        print("Using default config - this might cause issues!")
        configs = {
            'd_model': 128,
            'num_transformer_layers': 3,
            'num_head': 8
        }
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        print("Please check the model path!")
        return None
    
    # Create and run pipeline
    pipeline = CompletePredictionPipeline(
        model_path=model_path,
        configs=configs,
        base_data_dir="utils",  # Your utils directory
        save_dir="complete_alloy_predictions"
    )
    
    # Run the complete pipeline
    results = pipeline.run_complete_pipeline()
    
    if results is not None:
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📄 Final CSV shape: {results.shape}")
        print(f"📁 All results saved in: {pipeline.save_dir}")
        return results
    else:
        print(f"\n💥 Pipeline failed!")
        return None

if __name__ == '__main__':
    results = main()