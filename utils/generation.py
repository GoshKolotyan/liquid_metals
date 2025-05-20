from pandas import DataFrame
from itertools import combinations
import re
from collections import Counter

class GenData:
    """Generate alloy composite data"""

    def __init__(
        self, elements: list[str], min_fraction: float, max_element_in_composite: int
    ):
        self.elements = elements
        self.min_fraction = min_fraction
        self.max_element_in_composite = max_element_in_composite

    def _generate_component_fractions(self, number_of_components, min_fraction=0.1, step=0.1):
        """Generate component fractions that sum to 1.0 for any number of components."""
        if number_of_components <= 0:
            return []
            
        if number_of_components == 1:
            return [(1.0,)]
        
        valid_fractions = []
        possible_values = [round(i * step, 1) for i in range(
            int(min_fraction/step), 
            int((1.0 - (number_of_components-1)*min_fraction)/step) + 1
        )]
        
        if number_of_components == 2:
            for first in possible_values:
                second = round(1.0 - first, 1)
                if second >= min_fraction and second <= 0.9:
                    valid_fractions.append((first, second))
            return valid_fractions
            
        for first in possible_values:
            remaining = round(1.0 - first, 1)
            if remaining < min_fraction * (number_of_components - 1):
                continue  

            for sub_fractions in self._generate_sub_fractions(
                number_of_components - 1, 
                remaining, 
                min_fraction,
                step
            ):
                valid_fractions.append((first,) + sub_fractions)
                
        return valid_fractions
        
    def _generate_sub_fractions(self, num_components, total_sum, min_fraction, step):
        """Helper function to generate sub-fractions that sum to a given total."""
        if num_components == 1:
            if total_sum >= min_fraction and total_sum <= 0.9:
                return [(total_sum,)]
            return []
            
        valid_fractions = []
        max_for_this = total_sum - (num_components - 1) * min_fraction
        
        for fraction in [round(i * step, 1) for i in range(
            int(min_fraction/step),
            int(min(max_for_this, 0.9)/step) + 1
        )]:
            remaining = round(total_sum - fraction, 1)
            if remaining < min_fraction * (num_components - 1):
                continue
                
            for sub_fractions in self._generate_sub_fractions(
                num_components - 1,
                remaining,
                min_fraction,
                step
            ):
                valid_fractions.append((fraction,) + sub_fractions)
                
        return valid_fractions

    def generate_compositions(self, number_of_components: int, step=0.1):
        """Generate alloy compositions with the specified number of components."""
        alloys = set()
        compositions_data = []

        for elements in combinations(self.elements, number_of_components):
            for fractions in self._generate_component_fractions(number_of_components, self.min_fraction, step):
                composition = list(zip(elements, fractions))
                alloy = "".join([f"{element}{fraction:.1f}" for element, fraction in composition])
                alloys.add(alloy)
                
                compositions_data.append({
                    'alloy': alloy,
                    'elements': elements,
                    'fractions': fractions,
                    'num_components': number_of_components
                })
                
        return alloys, compositions_data

    def __str__(self):
        return "GenData"

    def __repr__(self):
        return f"Generating alloy composites for {self.elements=} in min ratio of {self.min_fraction=}"
    
    def analyze_compositions(self, all_composition_data):
        """Analyze the generated compositions to verify correctness"""
        df = DataFrame(all_composition_data)
        
        df['sum_fractions'] = df['fractions'].apply(lambda x: sum(x))
        fraction_sums_correct = all(abs(df['sum_fractions'] - 1.0) < 1e-10)
        print(f"All fraction sums are 1.0: {fraction_sums_correct}")
        
        df['min_fraction'] = df['fractions'].apply(lambda x: min(x))
        df['max_fraction'] = df['fractions'].apply(lambda x: max(x))
        min_fractions_correct = all(df['min_fraction'] >= self.min_fraction)
        max_fractions_correct = all(df['max_fraction'] <= 0.9)
        print(f"All fractions ≥ {self.min_fraction}: {min_fractions_correct}")
        print(f"All fractions ≤ 0.9: {max_fractions_correct}")
        
        component_counts = df['num_components'].value_counts().sort_index()
        print("\nCompositions by number of components:")
        for components, count in component_counts.items():
            print(f"  {components} components: {count} compositions")
        
        element_presence = {}
        for element in self.elements:
            element_presence[element] = sum(df['alloy'].apply(lambda x: element in x))
        
        print("\nElement presence in compositions:")
        for element, count in element_presence.items():
            print(f"  {element}: appears in {count} compositions ({count/len(df)*100:.1f}%)")
        
        all_fractions = [f for fractions in df['fractions'] for f in fractions]
        fraction_counts = Counter(all_fractions)
        
        print("\nFraction distribution:")
        for fraction in sorted(fraction_counts.keys()):
            print(f"  {fraction:.1f}: used {fraction_counts[fraction]} times")
        
        return df

    def __call__(self):
        output_ = []
        all_composition_data = []
        
        for i in range(2, self.max_element_in_composite + 1):
            print(f"Generating compositions with {i} components...")
            alloys, composition_data = self.generate_compositions(number_of_components=i, step=0.1)
            output_.extend(alloys)
            all_composition_data.extend(composition_data)
            print(f"  Found {len(alloys)} compositions")
        
        print(f"\nTotal compositions: {len(output_)}")
        self.df = DataFrame()
        self.df["System_Name"] = output_
        
        print("\n--- Sample of generated compositions ---")
        print(self.df.to_string())
        
        print("\n--- Analysis of generated compositions ---")
        self.analyze_compositions(all_composition_data)
        
        return self.df


# Example usage
cls = GenData(
    elements=["Ga", "In", "Sn", "Sb", "Bi"],
    min_fraction=0.1,
    max_element_in_composite=4,
)
cls()