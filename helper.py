import re 
import math
import json
import torch
from torch import Tensor

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
    def __init__(self, vocab_path="./Data/elements_vocab.json"):
        self.length = 5
        self.vocab = json.load(open(vocab_path))
        
    def calculate_entropy_of_mixing(self, elements_fractions):
        R = 8.314  # J/(mol*K)
        mix_entropy = 0.0
        for _, concentration in elements_fractions:
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
        avg = sum([self.vocab[element]['atomic radius'] * fraction for element, fraction in valid_elements])
        
        res = 0
        for element, fraction in valid_elements:
            ratio = self.vocab[element]['atomic radius'] / avg  # Fixed key name
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
            
        return math.sqrt(res)  # Remove the square root

    def __call__(self, elements_fractions: str) -> list:
        
        entropy_of_mixing = self.calculate_entropy_of_mixing(elements_fractions)
        mix_enthalpy = self.calculate_mix_enthalpy(elements_fractions)
        electronegativity_difference = self.calculate_electronegativity_difference(elements_fractions)
        atomic_radius_difference = self.calculate_atomic_radius_difference(elements_fractions)
        melting_point_difference = self.calculate_melting_point_difference(elements_fractions)
        print("Elements and fractions:", elements_fractions)
        print("Mixing enthalpy:", mix_enthalpy)
        print("Entropy of mixing:", entropy_of_mixing)
        print("Electronegativity difference:", electronegativity_difference)
        print("Atomic radius difference:", atomic_radius_difference)

        return [entropy_of_mixing, mix_enthalpy, electronegativity_difference, 
                atomic_radius_difference, melting_point_difference]

# Example usage
if __name__ == "__main__":
    feature_extractor = LM_feature_extractor()
    example = "Sn13.3Bi50Cd10Pb26.7"
    features = feature_extractor(example)
    print("Features:", features)