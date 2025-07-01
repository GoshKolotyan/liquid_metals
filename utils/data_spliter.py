import os 
import pandas as pd
from itertools import permutations
from parser import parse_compostion
from sklearn.model_selection import train_test_split
from collections import defaultdict

class DataSpliter:
    def __init__(self, data=None):
        if isinstance(data, pd.DataFrame):
            self.df = data
            self.data_path = None
        elif isinstance(data, str):
            self.data_path = data
            self.df = pd.read_csv(data)
        else:
            raise ValueError("data must be either a pandas DataFrame or a file path string")
        
        self.train_df = pd.DataFrame(columns=['System_Name','Tm (Liquidus)'])
        self.test_df = pd.DataFrame(columns=['System_Name','Tm (Liquidus)'])
        self.valid_df = pd.DataFrame(columns=['System_Name','Tm (Liquidus)'])
        self.train_ratio = 80  # percentage
        self.valid_ratio = 10  # percentage
        self.test_ratio = 10   # percentage

        self.usabel_elments = pd.read_json('./configs/elements_vocab.json').columns.tolist()
        print(f"Usable elements: {self.usabel_elments}")
        self.parser = parse_compostion
    
    def count_components(self, composition):
        parsed = self.parser(composition)
        parsed = [item for item in parsed if isinstance(item, tuple) and item[0] and item[1] != 0.0]
        return len(parsed)
    
    def augmenter(self, df=None):
        if df is None:
            df = self.df
            
        gens = []
        for index, row in df.iterrows():
            composition = row["System_Name"]
            try:
                target = float(row['Tm (Liquidus)'])
            except (ValueError, TypeError):
                continue

            parsed = self.parser(composition)
            parsed = [item for item in parsed if isinstance(item, tuple) and item[0] and item[1] != 0.0]
            
            for per in permutations(parsed):
                res = ''
                for comp, perc in per:
                    if comp not in self.usabel_elments:
                        res = ''
                        break
                    if perc > 1:
                        per /= 100
                    if perc == 1 or perc == 100:
                        res += f"{comp}"
                    else:
                        res += f"{comp}{perc}"
                    # res += f"{comp}" if perc == 1 else f"{comp}{perc}"
                gens.append({'System_Name': res, 'Tm (Liquidus)': target})

        return pd.DataFrame(gens)
    
    def categorize_by_component_count(self, df):
        component_dict = defaultdict(list)
        
        for idx, row in df.iterrows():
            composition = row["System_Name"]
            num_components = self.count_components(composition)
            component_dict[num_components].append(idx)
            
        return component_dict
    
    def stratified_split_by_components(self, random_state=42):
        """Split data ensuring all single-element compositions go to training
        and multi-element compositions are properly distributed"""
        augmented_df = self.augmenter(self.df)
        
        if augmented_df.empty:
            raise ValueError("No augmented data generated. Check your input data.")
        
        component_dict = self.categorize_by_component_count(augmented_df)
        
        # All single-element components go to training
        single_element_indices = component_dict.get(1, [])
        single_element_df = augmented_df.loc[single_element_indices].copy() if single_element_indices else pd.DataFrame(columns=augmented_df.columns)
        
        # Remove single elements from the dataframe for the regular split
        multi_element_df = augmented_df.drop(single_element_indices) if single_element_indices else augmented_df.copy()
        
        # Multi-component dictionaries for distributed splitting
        multi_component_dict = {k: v for k, v in component_dict.items() if k > 1}
        
        train_frames = [single_element_df] if not single_element_df.empty else []
        valid_frames = []
        test_frames = []
        
        for num_components, indices in multi_component_dict.items():
            component_df = multi_element_df.loc[indices]
            
            remaining_train_ratio = self.train_ratio - (len(single_element_df) / len(augmented_df) * 100)
            
            if remaining_train_ratio <= 0:
                valid_ratio_adjusted = self.valid_ratio / (self.valid_ratio + self.test_ratio)
                comp_valid_df, comp_test_df = train_test_split(
                    component_df,
                    test_size=(1 - valid_ratio_adjusted),
                    random_state=random_state,
                    shuffle=True
                )
                valid_frames.append(comp_valid_df)
                test_frames.append(comp_test_df)
            else:
                train_ratio_adjusted = remaining_train_ratio / (remaining_train_ratio + self.valid_ratio + self.test_ratio)
                
                # First split into train and temp (valid+test)
                comp_train_df, temp_df = train_test_split(
                    component_df,
                    test_size=(1 - train_ratio_adjusted),
                    random_state=random_state,
                    shuffle=True
                )
                
                # Then split temp into valid and test
                valid_ratio_adjusted = self.valid_ratio / (self.valid_ratio + self.test_ratio)
                comp_valid_df, comp_test_df = train_test_split(
                    temp_df,
                    test_size=(1 - valid_ratio_adjusted),
                    random_state=random_state,
                    shuffle=True
                )
                
                # Add to our list of frames
                train_frames.append(comp_train_df)
                valid_frames.append(comp_valid_df)
                test_frames.append(comp_test_df)
        
        self.train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame(columns=augmented_df.columns)
        self.valid_df = pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame(columns=augmented_df.columns)
        self.test_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame(columns=augmented_df.columns)
        
        return self.train_df, self.valid_df, self.test_df
    
    def get_split_info(self, include_component_distribution=True):
        total_size = len(self.train_df) + len(self.valid_df) + len(self.test_df)
        
        info = {
            'train_size': len(self.train_df),
            'valid_size': len(self.valid_df),
            'test_size': len(self.test_df),
            'total_size': total_size
        }
        
        if total_size > 0:
            info['train_percentage'] = (len(self.train_df) / total_size) * 100
            info['valid_percentage'] = (len(self.valid_df) / total_size) * 100
            info['test_percentage'] = (len(self.test_df) / total_size) * 100
        
        if include_component_distribution:
            train_components = self.categorize_by_component_count(self.train_df)
            valid_components = self.categorize_by_component_count(self.valid_df)
            test_components = self.categorize_by_component_count(self.test_df)
            
            original_components = self.categorize_by_component_count(self.df)
            augmented_components = self.categorize_by_component_count(self.augmenter())
            
            info['component_distribution'] = {
                'train': {k: len(v) for k, v in train_components.items()},
                'valid': {k: len(v) for k, v in valid_components.items()},
                'test': {k: len(v) for k, v in test_components.items()}
            }
            
            info['original_composition_counts'] = {
                k: len(v) for k, v in original_components.items()
            }
            
            info['augmented_composition_counts'] = {
                k: len(v) for k, v in augmented_components.items()
            }
        
        return info
    
    def save_description(self, output_dir, file_name="description.txt"):
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, file_name)
        
        split_info = self.get_split_info(include_component_distribution=True)
        
        original_components = self.categorize_by_component_count(self.df)
        
        augmented_df = self.augmenter()
        augmented_components = self.categorize_by_component_count(augmented_df)
        
        with open(file_path, 'w') as f:
            f.write("=================================================\n")
            f.write("          DATASET DESCRIPTION & SPLITTING         \n")
            f.write("=================================================\n\n")
            
            f.write("ORIGINAL DATASET INFORMATION\n")
            f.write("--------------------------\n")
            f.write(f"Total number of compositions: {len(self.df)}\n\n")
            
            f.write("Composition counts by component number:\n")
            for n_comp, indices in sorted(original_components.items()):
                label = "single" if n_comp == 1 else "double" if n_comp == 2 else "triple" if n_comp == 3 else "quadruple"
                f.write(f"{n_comp}({label}) - number of compositions found: {len(indices)}\n")
                
                if indices:
                    f.write("First 5 sample compositions:\n")
                    for i, idx in enumerate(indices[:5]):
                        f.write(f"  {i+1}. {self.df.iloc[idx]['System_Name']} - {self.df.iloc[idx]['Tm (Liquidus)']}\n")
                f.write("\n")
            
            f.write("\nAUGMENTATION INFORMATION\n")
            f.write("-----------------------\n")
            f.write(f"Total number of compositions after augmentation: {len(augmented_df)}\n\n")
            
            f.write("Augmented composition counts by component number:\n")
            for n_comp, indices in sorted(augmented_components.items()):
                label = "single" if n_comp == 1 else "double" if n_comp == 2 else "triple" if n_comp == 3 else "quadruple"
                f.write(f"{n_comp}({label}) - number of compositions found: {len(indices)}\n")
                
                if indices:
                    f.write("First 5 sample compositions:\n")
                    for i, idx in enumerate(indices[:5]):
                        f.write(f"  {i+1}. {augmented_df.iloc[idx]['System_Name']} - {augmented_df.iloc[idx]['Tm (Liquidus)']}\n")
                f.write("\n")
            
            f.write("\nSPLIT INFORMATION\n")
            f.write("----------------\n")
            f.write(f"Train size: {split_info['train_size']} compositions ({split_info['train_percentage']:.2f}%)\n")
            f.write(f"Validation size: {split_info['valid_size']} compositions ({split_info['valid_percentage']:.2f}%)\n")
            f.write(f"Test size: {split_info['test_size']} compositions ({split_info['test_percentage']:.2f}%)\n")
            f.write(f"Total size: {split_info['total_size']} compositions\n\n")
            
            f.write("Component distribution after splitting:\n")
            component_dist = split_info['component_distribution']
            
            f.write("\nNumber of compositions by component count:\n")
            max_components = max(
                max(component_dist['train'].keys(), default=0),
                max(component_dist['valid'].keys(), default=0),
                max(component_dist['test'].keys(), default=0)
            )
            
            f.write(f"{'Components':<12} {'Label':<10} {'Train':<10} {'Valid':<10} {'Test':<10}\n")
            f.write("-" * 60 + "\n")
            
            for n in range(1, max_components + 1):
                label = "single" if n == 1 else "double" if n == 2 else "triple" if n == 3 else "quadruple"
                train_count = component_dist['train'].get(n, 0)
                valid_count = component_dist['valid'].get(n, 0)
                test_count = component_dist['test'].get(n, 0)
                f.write(f"{n:<12} {label:<10} {train_count:<10} {valid_count:<10} {test_count:<10}\n")
            
            f.write("\n\nDATA SPLITTING METHODOLOGY\n")
            f.write("-------------------------\n")
            f.write("The data was split using a component-stratified approach with the following characteristics:\n")
            f.write("1. All single-element compositions were included in the training set to ensure the model\n")
            f.write("   learns the base properties of each element.\n")
            f.write("2. Multi-element compositions (double, triple, quadruple) were distributed across\n")
            f.write("   training, validation, and test sets following the 80/10/10 ratio.\n")
            f.write("3. The split was stratified by component count to ensure proper representation\n")
            f.write("   of each component category in each split.\n")
            f.write("4. Data augmentation was performed by generating all possible permutations of\n")
            f.write("   the chemical compositions to increase the dataset size and diversity.\n\n")
            
        
        print(f"Detailed description saved to: {file_path}")
        return file_path

    def __call__(self, 
                    output_dir, 
                    random_state, 
                    save, 
                    verbose):
            """Execute the complete pipeline with stratified component splitting"""
            if verbose:
                print("Starting data augmentation and component-stratified splitting pipeline...")
                print(f"Original data size: {len(self.df)} rows")
                
                original_components = self.categorize_by_component_count(self.df)
                print("\nOriginal composition counts by component number:")
                for n_comp, indices in sorted(original_components.items()):
                    label = "single" if n_comp == 1 else "double" if n_comp == 2 else "triple" if n_comp == 3 else "quadruple"
                    print(f"{n_comp}({label}) - number of compositions found: {len(indices)}")
                    # Print first 5 compositions for this component count
                    if indices:
                        print("First 5 sample compositions:")
                        for i, idx in enumerate(indices[:2]):
                            print(f"  {i+1}. {self.df.iloc[idx]['System_Name']} - {self.df.iloc[idx]['Tm (Liquidus)']}")
                    # print()
                
            
            train_df, valid_df, test_df = self.stratified_split_by_components(random_state=random_state)
            
            if verbose:
                print("\nSplit information:")
                split_info = self.get_split_info(include_component_distribution=True)
                
                for key, value in split_info.items():
                    if key not in ['component_distribution', 'original_composition_counts', 'augmented_composition_counts']:
                        if 'percentage' in key:
                            print(f"{key}: {value:.2f}%")
                        else:
                            print(f"{key}: {value}")
                
                print("\nComponent distribution after splitting:")
                component_dist = split_info['component_distribution']
                
                print("\nNumber of compositions by component count:")
                max_components = max(
                    max(component_dist['train'].keys(), default=0),
                    max(component_dist['valid'].keys(), default=0),
                    max(component_dist['test'].keys(), default=0)
                )
                
                print(f"{'Components':<12} {'Label':<10} {'Train':<10} {'Valid':<10} {'Test':<10}")
                print("-" * 60)
                
                for n in range(1, max_components + 1):
                    label = "single" if n == 1 else "double" if n == 2 else "triple" if n == 3 else "quadruple"
                    train_count = component_dist['train'].get(n, 0)
                    valid_count = component_dist['valid'].get(n, 0)
                    test_count = component_dist['test'].get(n, 0)
                    print(f"{n:<12} {label:<10} {train_count:<10} {valid_count:<10} {test_count:<10}")
                
            
            if save:
                os.makedirs(output_dir, exist_ok=True)
                self.train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
                self.valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
                self.test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
                
                self.save_description(output_dir)
                
                if verbose:
                    print(f"\nData saved to: {output_dir}")
            
            if verbose:
                print("\nPipeline completed successfully!")
            
            return self.train_df, self.valid_df, self.test_df


if __name__ == "__main__":
    example_df = pd.read_csv('Data/All_Data_Based_06-06-2025/13-06-2025-fixed.csv')
    
    splitter = DataSpliter(example_df)
    

    train, valid, test = splitter(
        output_dir='./Data/Component_Stratified_Split_Based_on_Augmentation_28_06_2025',
        random_state=123,
        save=True,
        verbose=True
    )
    
    augmented = splitter.augmenter()
    print(f"Total augmented rows: {len(augmented)}")