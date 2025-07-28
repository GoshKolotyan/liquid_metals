from pandas import DataFrame
from itertools import combinations
import pandas as pd
import numpy as np
import os

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
            # Triplet: step size 0.1
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
            # Quadruplet: step size 0.1
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
                            
        elif num_elements == 5:
            # Quintuplet: step size 0.2
            ratios = np.arange(step_size, 1.0, step_size)
            for ratio1 in ratios:
                for ratio2 in ratios:
                    for ratio3 in ratios:
                        for ratio4 in ratios:
                            ratio5 = 1.0 - ratio1 - ratio2 - ratio3 - ratio4
                            if ratio5 >= step_size and abs(ratio1 + ratio2 + ratio3 + ratio4 + ratio5 - 1.0) < 1e-10:
                                # Format ratios
                                ratio1_str = f"{ratio1:.1f}".rstrip('0').rstrip('.')
                                ratio2_str = f"{ratio2:.1f}".rstrip('0').rstrip('.')
                                ratio3_str = f"{ratio3:.1f}".rstrip('0').rstrip('.')
                                ratio4_str = f"{ratio4:.1f}".rstrip('0').rstrip('.')
                                ratio5_str = f"{ratio5:.1f}".rstrip('0').rstrip('.')
                                
                                alloy_name = f"{elements_tuple[0]}{ratio1_str}{elements_tuple[1]}{ratio2_str}{elements_tuple[2]}{ratio3_str}{elements_tuple[3]}{ratio4_str}{elements_tuple[4]}{ratio5_str}"
                                combinations_list.append(alloy_name)
        
        return combinations_list

    def generate_combinations(self, number_of_components: int):
        """Generate all element combinations with ratios for the specified number of components."""
        combinations_list = []
        
        # Define step sizes based on number of components
        step_sizes = {3: 0.1, 4: 0.1, 5: 0.1}
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

    def extract_elements_from_name(self, system_name):
        """Extract element names from a system name like 'Ga0.3In0.3Sn0.4' -> ['Ga', 'In', 'Sn']"""
        elements = []
        i = 0
        while i < len(system_name):
            if system_name[i].isupper():
                element = system_name[i]
                i += 1
                # Check if there's a lowercase letter (for elements like 'Bi', 'Sb')
                if i < len(system_name) and system_name[i].islower():
                    element += system_name[i]
                    i += 1
                elements.append(element)
                # Skip the ratio part
                while i < len(system_name) and not system_name[i].isupper():
                    i += 1
            else:
                i += 1
        return elements

    def save_by_element_combinations(self):
        """Save each unique element combination to separate CSV files organized by component count"""
        if not hasattr(self, 'df') or self.df.empty:
            print("No data to save. Please run the generation first.")
            return
        
        # Define folder names for different component counts
        folder_names = {3: 'ternary', 4: 'quaternary', 5: 'pentanary'}
        
        # Create folders if they don't exist
        for folder in folder_names.values():
            os.makedirs(folder, exist_ok=True)
        
        # Group combinations by their element composition
        element_groups = {}
        
        for system_name in self.df['System_Name']:
            elements = self.extract_elements_from_name(system_name)
            num_components = len(elements)
            # Create a sorted key for consistent grouping
            element_key = ''.join(sorted(elements))
            
            if element_key not in element_groups:
                element_groups[element_key] = {'combinations': [], 'num_components': num_components}
            element_groups[element_key]['combinations'].append(system_name)
        
        # Save each group to its own CSV file in the appropriate folder
        print("\n--- Saving element combinations to organized folders ---")
        
        folder_counts = {3: 0, 4: 0, 5: 0}
        
        for element_key, group_data in element_groups.items():
            combinations = group_data['combinations']
            num_components = group_data['num_components']
            
            if num_components in folder_names:
                folder_name = folder_names[num_components]
                filename = os.path.join(folder_name, f"{element_key}.csv")
                
                group_df = DataFrame({'System_Name': combinations})
                group_df.to_csv(filename, index=False)
                
                folder_counts[num_components] += 1
                print(f"{element_key} ({num_components} components) → {filename} ({len(combinations)} combinations)")
        
        print(f"\n--- Summary ---")
        for components, count in folder_counts.items():
            if count > 0:
                folder_name = folder_names[components]
                print(f"{folder_name}/ folder: {count} element combinations")
        
        print(f"Total unique element combinations: {sum(folder_counts.values())}")

    def __call__(self):
        all_combinations = []
        self.component_dfs = {}  # Store separate DataFrames for each component type
        
        # Only generate 3, 4, and 5 component combinations
        for i in range(3, self.max_element_in_composite + 1):
            print(f"Generating combinations with {i} components...")
            combinations_list = self.generate_combinations(number_of_components=i)
            all_combinations.extend(combinations_list)
            print(f"  Found {len(combinations_list)} combinations")
            
            # Store each component type separately
            component_df = DataFrame({'System_Name': combinations_list})
            self.component_dfs[i] = component_df
        
        print(f"\nTotal combinations: {len(all_combinations)}")
        self.df = DataFrame()
        self.df["System_Name"] = all_combinations
        
        print("\n--- Sample of generated combinations ---")
        print(self.df.head(15).to_string())  # Show first 15 combinations
        
        print("\n--- Analysis of generated combinations ---")
        self.analyze_combinations(all_combinations)
        
        return self.df


# Example usage with 5 components
example_elements = [
    "Ga", "In", "Sn", "Sb", "Bi"
]  # Small set for testing

# Create instance with max 5 components
cls = GenData(
    elements=example_elements,
    max_element_in_composite=5,  # Now includes 5-component combinations
)

# Generate combinations with ratios (3, 4, and 5 components only)
result_df = cls()

# Save each unique element combination to separate CSV files
cls.save_by_element_combinations()

# Display examples of each component type
print("\n--- Examples by component count ---")
for i in range(3, 6):  # Only 3, 4, and 5 components
    if i in cls.component_dfs:
        df = cls.component_dfs[i]
        print(f"\n{i}-component examples:")
        print(df.head(3)['System_Name'].tolist())

# Show detailed summary
print(f"\n--- Summary by component count ---")
for i in range(3, 6):
    if i in cls.component_dfs:
        count = len(cls.component_dfs[i])
        print(f"{i} components: {count} combinations")
print(f"Total: {len(result_df)} combinations")

# Show some examples of element combination files that will be created
print(f"\n--- Example folder structure ---")
sample_combinations = result_df['System_Name'].head(15).tolist()
folder_examples = {3: set(), 4: set(), 5: set()}

for combo in sample_combinations:
    elements = cls.extract_elements_from_name(combo)
    element_key = ''.join(sorted(elements))
    num_components = len(elements)
    
    if num_components in folder_examples:
        folder_examples[num_components].add(element_key)

folder_names = {3: 'ternary', 4: 'quaternary', 5: 'pentanary'}
for components in [3, 4, 5]:
    if folder_examples[components]:
        folder_name = folder_names[components]
        print(f"\n{folder_name}/ folder:")
        for element_combo in sorted(list(folder_examples[components])[:3]):  # Show first 3 examples
            print(f"  {element_combo}.csv")