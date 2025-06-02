from pandas import DataFrame
from itertools import combinations
import pandas as pd
import numpy as np

class GenData:
    """Generate alloy element combinations with ratio logic"""

    def __init__(self, elements: list[str], max_element_in_composite: int):
        self.elements = elements
        self.max_element_in_composite = max_element_in_composite

    def generate_ratio_combinations(self, elements_tuple, step_size):
        """Generate all ratio combinations for given elements with specified step size"""
        num_elements = len(elements_tuple)
        combinations_list = []
        
        if num_elements == 2:
            # Binary: step size 0.01, range from 0.01 to 0.99
            ratios = np.arange(step_size, 1.0, step_size)
            for ratio1 in ratios:
                ratio2 = 1.0 - ratio1
                # Format ratios to avoid floating point precision issues
                ratio1_str = f"{ratio1:.2f}".rstrip('0').rstrip('.')
                ratio2_str = f"{ratio2:.2f}".rstrip('0').rstrip('.')
                
                alloy_name = f"{elements_tuple[0]}{ratio1_str}{elements_tuple[1]}{ratio2_str}"
                combinations_list.append(alloy_name)
                
        elif num_elements == 3:
            # Triplet: step size 0.2
            ratios = np.arange(step_size, 1.0, step_size)
            for ratio1 in ratios:
                for ratio2 in ratios:
                    ratio3 = 1.0 - ratio1 - ratio2
                    if ratio3 >= step_size and abs(ratio1 + ratio2 + ratio3 - 1.0) < 1e-10:
                        # Format ratios
                        ratio1_str = f"{ratio1:.1f}".rstrip('0').rstrip('.')
                        ratio2_str = f"{ratio2:.1f}".rstrip('0').rstrip('.')
                        ratio3_str = f"{ratio3:.1f}".rstrip('0').rstrip('.')
                        
                        alloy_name = f"{elements_tuple[0]}{ratio1_str}{elements_tuple[1]}{ratio2_str}{elements_tuple[2]}{ratio3_str}"
                        combinations_list.append(alloy_name)
                        
        elif num_elements == 4:
            # Quadruplet: step size 0.2
            ratios = np.arange(step_size, 1.0, step_size)
            for ratio1 in ratios:
                for ratio2 in ratios:
                    for ratio3 in ratios:
                        ratio4 = 1.0 - ratio1 - ratio2 - ratio3
                        if ratio4 >= step_size and abs(ratio1 + ratio2 + ratio3 + ratio4 - 1.0) < 1e-10:
                            # Format ratios
                            ratio1_str = f"{ratio1:.1f}".rstrip('0').rstrip('.')
                            ratio2_str = f"{ratio2:.1f}".rstrip('0').rstrip('.')
                            ratio3_str = f"{ratio3:.1f}".rstrip('0').rstrip('.')
                            ratio4_str = f"{ratio4:.1f}".rstrip('0').rstrip('.')
                            
                            alloy_name = f"{elements_tuple[0]}{ratio1_str}{elements_tuple[1]}{ratio2_str}{elements_tuple[2]}{ratio3_str}{elements_tuple[3]}{ratio4_str}"
                            combinations_list.append(alloy_name)
        
        return combinations_list

    def generate_combinations(self, number_of_components: int):
        """Generate all element combinations with ratios for the specified number of components."""
        combinations_list = []
        
        # Define step sizes based on number of components
        step_sizes = {2: 0.01, 3: 0.1, 4: 0.1}
        step_size = step_sizes.get(number_of_components, 0.1)
        
        for elements_tuple in combinations(self.elements, number_of_components):
            ratio_combinations = self.generate_ratio_combinations(elements_tuple, step_size)
            combinations_list.extend(ratio_combinations)
                
        return combinations_list

    def __str__(self):
        return "GenData with Ratio Logic"

    def __repr__(self):
        return f"Generating element combinations with ratios for {self.elements=}"
    
    def analyze_combinations(self, all_combinations):
        """Analyze the generated combinations"""
        df = DataFrame({'System_Name': all_combinations})
        
        # Extract number of components by counting elements (more complex due to ratios)
        def count_elements(system_name):
            # Count uppercase letters (assuming element names start with uppercase)
            return sum(1 for c in system_name if c.isupper())
        
        df['num_components'] = df['System_Name'].apply(count_elements)
        component_counts = df['num_components'].value_counts().sort_index()
        
        print("Combinations by number of components:")
        for components, count in component_counts.items():
            print(f"  {components} components: {count} combinations")
        
        # Element presence analysis
        element_presence = {}
        for element in self.elements:
            count = sum(df['System_Name'].apply(lambda x: element in x))
            element_presence[element] = count
        
        print("\nElement presence in combinations:")
        for element, count in sorted(element_presence.items(), key=lambda x: x[1], reverse=True):
            print(f"  {element}: appears in {count} combinations ({count/len(df)*100:.1f}%)")
        
        return df

    def save_to_csv(self, filename="element_combinations_with_ratios.csv"):
        """Save the generated combinations to a CSV file"""
        if hasattr(self, 'df') and not self.df.empty:
            self.df.to_csv(filename, index=False)
            print(f"\nCombinations saved to {filename}")
        else:
            print("No data to save. Please run the generation first.")

    def __call__(self):
        all_combinations = []
        
        for i in range(2, self.max_element_in_composite + 1):
            print(f"Generating combinations with {i} components...")
            combinations_list = self.generate_combinations(number_of_components=i)
            all_combinations.extend(combinations_list)
            print(f"  Found {len(combinations_list)} combinations")
        
        print(f"\nTotal combinations: {len(all_combinations)}")
        self.df = DataFrame()
        self.df["System_Name"] = all_combinations
        
        print("\n--- Sample of generated combinations ---")
        print(self.df.head(15).to_string())  # Show first 15 combinations
        
        print("\n--- Analysis of generated combinations ---")
        self.analyze_combinations(all_combinations)
        
        return self.df


# Example usage with a smaller set of elements for demonstration
# You can replace this with your actual elements loading
example_elements = [
    "Ga", "In", "Sn", "Sb", "Bi"
]  # Small set for testing

# If you want to load from JSON (uncomment and modify path as needed):
# elements = pd.read_json("../configs/elements_vocab.json")
# element_list = elements.columns.tolist()

cls = GenData(
    elements=example_elements,  # Use example_elements or element_list
    max_element_in_composite=4,
)

# Generate combinations with ratios
result_df = cls()

# Save to CSV
cls.save_to_csv("element_combinations_with_ratios.csv")

# Display some examples of each type
print("\n--- Examples by component count ---")
for i in range(2, 5):
    subset = result_df[result_df['System_Name'].apply(lambda x: sum(1 for c in x if c.isupper()) == i)]
    if not subset.empty:
        print(f"\n{i}-component examples:")
        print(subset.head(5)['System_Name'].tolist())