import os
import re 
import math
import json
import torch
import pytorch_lightning as pl

def get_next_version(logs_dir, model_name, model_version_prefix):
    """Determine the next available version number for a given model version prefix."""
    # Create the full path to check
    base_path = os.path.join(logs_dir, model_name)
    
    if not os.path.exists(base_path):
        return f"{model_version_prefix}.1"
    
    # Get all folders that match the version prefix
    matching_versions = []
    for item in os.listdir(base_path):
        if item.startswith(model_version_prefix + "_") and os.path.isdir(os.path.join(base_path, item)):
            try:
                # Extract the version number after the prefix and underscore
                version_num = int(item.split("_")[1])
                matching_versions.append(version_num)
            except (IndexError, ValueError):
                continue
    
    # If no matching versions found, start with 1
    if not matching_versions:
        return f"{model_version_prefix}.1"
    
    # Otherwise, increment the highest version number
    next_version = max(matching_versions) + 1
    return f"{model_version_prefix}.{next_version}"


def save_final_model(model, config, trainer, best_checkpoint, timestamp, checkpoint_dir, test_results=None):
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'epoch': trainer.current_epoch,
        'global_step': trainer.global_step,
        'best_val_loss': best_checkpoint.best_model_score.item() if best_checkpoint.best_model_score else None,
        'timestamp': timestamp,
        'version': config.model_version,
        'pytorch_version': torch.__version__,
        'pytorch_lightning_version': pl.__version__,
        'test_results': test_results,
        'description': config.description
    }, final_model_path)
    
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_state_dict.pt"))
    
    return final_model_path

def save_config(config, checkpoint_dir):
    config_dict = config.as_dict()
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)


def create_model_loader(checkpoint_dir, config, model_path):
    """Creates a helper script to easily load this exact model"""
    script_content = f"""
import torch
import sys
import os
import json
from pathlib import Path

# Add parent directory to Python path to find the model module
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

# Import the model class - update the import path as needed
from model import ChemicalTransformerLightning

def load_model(device=None):
    '''
    Load the saved model
    
    Args:
        device: torch device to load the model to (default: None, uses CUDA if available)
        
    Returns:
        model: The loaded model
        config: Model configuration
    '''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved checkpoint
    checkpoint = torch.load('{model_path}', map_location=device)
    
    # Extract the config
    config = checkpoint['config']
    
    # Create a model with the same configuration
    model = ChemicalTransformerLightning(config)
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    print(f"Model loaded successfully from checkpoint created at epoch {{checkpoint['epoch']}}")
    print(f"Best validation loss: {{checkpoint.get('best_val_loss', 'Not recorded')}}")
    
    return model, config

def predict(model, input_data):
    '''
    Make a prediction using the loaded model
    
    Args:
        model: The loaded model
        input_data: Input data for prediction (format depends on model requirements)
        
    Returns:
        Prediction result
    '''
    # Set model to evaluation mode
    model.eval()
    
    # Perform prediction (implementation depends on your model's interface)
    with torch.no_grad():
        prediction = model(input_data)
    
    return prediction

if __name__ == "__main__":
    # Example usage
    model, config = load_model()
    print("\\nModel configuration:")
    for key, value in config.items():
        print(f"  {{key}}: {{value}}")
    
    print("\\nTo use this model for prediction, use the predict() function.")
    print("Example: prediction = predict(model, your_input_data)")
"""
    
    with open(f"{checkpoint_dir}/load_model.py", 'w') as f:
        f.write(script_content)
        
    print(f"Created model loader script: {checkpoint_dir}/load_model.py")


def parse_compostion(composition: str) -> list[tuple[str, float]]:
    pattern = r'([A-Z][a-z]*)([0-9]*\.?[0-9]*)'
    matches = re.findall(pattern, composition)

    elements_fractions = []
    for element, fraction in matches:
        # print(f"Element: {element}, Fraction: {fraction}")
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
    
    while len(elements_fractions) < 4:
        elements_fractions.append(("", 0.0))
    return elements_fractions

class LM_feature_extractor:
    def __init__(self, vocab_path="./configs/elements_vocab.json"):
        self.length = 5
        self.vocab = json.load(open(vocab_path))
        
    def calculate_entropy_of_mixing(self, elements_fractions):
        R = 8.314  # J/(mol*K)
        mix_entropy = 0.0
        for _, concentration in elements_fractions:
            if concentration > 0:
                mix_entropy += concentration * math.log(concentration)
        mix_entropy = -R * mix_entropy
        return mix_entropy
        
    def calculate_mix_enthalpy(self, elements_fractions):
        # Placeholder for future implementation
        return 0.0

    def calculate_electronegativity_difference(self, elements_fractions):
        # Filter out empty elements
        valid_elements = [(element, fraction) for element, fraction in elements_fractions if element and fraction > 0]
        
        # Calculate average electronegativity
        avg = sum([self.vocab[element]['electronegativity'] * fraction for element, fraction in valid_elements])
        
        # Calculate electronegativity difference
        res = 0
        for element, fraction in valid_elements:
            ratio = self.vocab[element]['electronegativity'] / avg
            term = fraction * (1 - ratio)**2
            res += term
        
        return math.sqrt(res)  # Remove the square root as per the formula

    def calculate_atomic_radius_difference(self, elements_fractions):
        # Filter out empty elements
        valid_elements = [(element, fraction) for element, fraction in elements_fractions if element and fraction > 0]
        
        # Calculate average atomic radius
        avg = sum([self.vocab[element]['atomic_radius'] * fraction for element, fraction in valid_elements])
        
        res = 0
        for element, fraction in valid_elements:
            ratio = self.vocab[element]['atomic_radius'] / avg  # Fixed key name
            term = fraction * (1 - ratio)**2
            res += term
            
        return math.sqrt(res)  # Remove the square root

    def calculate_melting_point_difference(self, elements_fractions):
        # Filter out empty elements
        valid_elements = [(element, fraction) for element, fraction in elements_fractions if element and fraction > 0]
        
        # Calculate average melting point
        avg = sum([self.vocab[element]['melting_point'] * fraction for element, fraction in valid_elements])
        
        res = 0
        for element, fraction in valid_elements:
            ratio = self.vocab[element]['melting_point'] / avg
            term = fraction * (1 - ratio)**2
            res += term
            
        return math.sqrt(res)  
    
    def __call__(self, elements_fractions: str) -> list:
        
        entropy_of_mixing = self.calculate_entropy_of_mixing(elements_fractions)
        mix_enthalpy = self.calculate_mix_enthalpy(elements_fractions)
        electronegativity_difference = self.calculate_electronegativity_difference(elements_fractions)
        atomic_radius_difference = self.calculate_atomic_radius_difference(elements_fractions)
        melting_point_difference = self.calculate_melting_point_difference(elements_fractions)
        # print("Elements and fractions:", elements_fractions)
        # print("Mixing enthalpy:", mix_enthalpy)
        # print("Entropy of mixing:", entropy_of_mixing)
        # print("Electronegativity difference:", electronegativity_difference)
        # print("Atomic radius difference:", atomic_radius_difference)

        return [entropy_of_mixing, mix_enthalpy, electronegativity_difference, 
                atomic_radius_difference, melting_point_difference]

# # Example usage
# if __name__ == "__main__":
#     feature_extractor = LM_feature_extractor()
#     example = "Sn13.3Bi50Cd10Pb26.7"
#     features = feature_extractor(example)
#     print("Features:", features)