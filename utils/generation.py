from pandas import DataFrame
from itertools import combinations
import pandas as pd

class GenData:
    """Generate alloy element combinations"""

    def __init__(self, elements: list[str], max_element_in_composite: int):
        self.elements = elements
        self.max_element_in_composite = max_element_in_composite

    def generate_combinations(self, number_of_components: int):
        """Generate all element combinations with the specified number of components."""
        combinations_list = []
        
        for elements in combinations(self.elements, number_of_components):
            # Create alloy name by joining elements with hyphens
            alloy_name = "-".join(elements)
            combinations_list.append(alloy_name)
                
        return combinations_list

    def __str__(self):
        return "GenData"

    def __repr__(self):
        return f"Generating element combinations for {self.elements=}"
    
    def analyze_combinations(self, all_combinations):
        """Analyze the generated combinations"""
        df = DataFrame({'System_Name': all_combinations})
        
        # Count combinations by number of components (count hyphens + 1)
        df['num_components'] = df['System_Name'].apply(lambda x: x.count('-') + 1)
        component_counts = df['num_components'].value_counts().sort_index()
        
        print("Combinations by number of components:")
        for components, count in component_counts.items():
            print(f"  {components} components: {count} combinations")
        
        # Element presence analysis
        element_presence = {}
        for element in self.elements:
            element_presence[element] = sum(df['System_Name'].apply(lambda x: element in x))
        
        print("\nElement presence in combinations:")
        for element, count in element_presence.items():
            print(f"  {element}: appears in {count} combinations ({count/len(df)*100:.1f}%)")
        
        return df

    def save_to_csv(self, filename="element_combinations.csv"):
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
        print(self.df.head(10).to_string())  # Show first 10 combinations
        
        print("\n--- Analysis of generated combinations ---")
        self.analyze_combinations(all_combinations)
        
        return self.df


# Load elements from JSON file
elements = pd.read_json("../configs/elements_vocab.json")

# Example usage
cls = GenData(
    elements=elements.columns.tolist(),
    max_element_in_composite=4,
)

# Generate combinations
result_df = cls()

# Save to CSV
cls.save_to_csv("element_combinations.csv")

# Optional: Save with custom filename
# cls.save_to_csv("my_alloy_combinations.csv")