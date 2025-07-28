import pandas as pd
import numpy as np
import os
from collections import defaultdict
from itertools import permutations
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from parser import parse_compostion

class DataSpliter:
    def __init__(self, data=None, apply_augmentation=True):
        """
        Initialize DataSplitter with optional augmentation control.
        
        Args:
            data: pandas DataFrame or file path string
            apply_augmentation: bool, whether to apply permutation augmentation (default: True)
        """
        if isinstance(data, pd.DataFrame):
            self.df = data
            self.data_path = None
        elif isinstance(data, str):
            self.data_path = data
            self.df = pd.read_csv(data)
        else:
            raise ValueError("data must be either a pandas DataFrame or a file path string")
        
        # 🚀 NEW: Clean and validate data
        self.df = self._clean_and_validate_data(self.df)
        
        self.train_df = pd.DataFrame(columns=['System_Name','Tm (Liquidus)'])
        self.valid_df = pd.DataFrame(columns=['System_Name','Tm (Liquidus)'])
        self.train_ratio = 85  # percentage
        self.valid_ratio = 15  # percentage

        # 🚀 NEW: Augmentation control flag
        self.apply_augmentation = apply_augmentation
        self.augmentation_applied = False  # Track if augmentation was actually applied

        self.usabel_elments = pd.read_json('./configs/elements_vocab.json').columns.tolist()
        print(f"Usable elements: {self.usabel_elments}")
        print(f"🔧 Augmentation enabled: {self.apply_augmentation}")
        self.parser = parse_compostion
    
    def _clean_and_validate_data(self, df):
        """
        🚀 NEW: Clean and validate the input data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Cleaned and validated DataFrame
        """
        print("🧹 Cleaning and validating data...")
        original_size = len(df)
        
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # 1. Remove rows with missing System_Name or Tm (Liquidus)
        cleaned_df = cleaned_df.dropna(subset=['System_Name', 'Tm (Liquidus)'])
        
        # 2. Remove rows with empty strings
        cleaned_df = cleaned_df[cleaned_df['System_Name'].str.strip() != '']
        cleaned_df = cleaned_df[cleaned_df['Tm (Liquidus)'].astype(str).str.strip() != '']
        
        # 3. Try to convert Tm (Liquidus) to float and remove invalid entries
        def is_valid_float(x):
            try:
                float(x)
                return True
            except (ValueError, TypeError):
                return False
        
        valid_mask = cleaned_df['Tm (Liquidus)'].apply(is_valid_float)
        invalid_rows = cleaned_df[~valid_mask]
        
        if len(invalid_rows) > 0:
            print(f"⚠️  Found {len(invalid_rows)} rows with invalid target values:")
            for idx, row in invalid_rows.head(5).iterrows():
                print(f"   Row {idx}: '{row['System_Name']}' -> '{row['Tm (Liquidus)']}' (invalid)")
            if len(invalid_rows) > 5:
                print(f"   ... and {len(invalid_rows) - 5} more")
        
        cleaned_df = cleaned_df[valid_mask]
        
        # 4. Convert Tm (Liquidus) to float
        cleaned_df['Tm (Liquidus)'] = cleaned_df['Tm (Liquidus)'].astype(float)
        
        # 5. Remove duplicate compositions (keep first occurrence)
        initial_size = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=['System_Name'], keep='first')
        duplicates_removed = initial_size - len(cleaned_df)
        
        if duplicates_removed > 0:
            print(f"🔄 Removed {duplicates_removed} duplicate compositions")
        
        # 6. Validate compositions can be parsed
        def can_parse_composition(comp):
            try:
                parsed = self.parser(comp) if hasattr(self, 'parser') else []
                return len(parsed) > 0
            except:
                return False
        
        if hasattr(self, 'parser'):
            parseable_mask = cleaned_df['System_Name'].apply(can_parse_composition)
            unparseable_rows = cleaned_df[~parseable_mask]
            
            if len(unparseable_rows) > 0:
                print(f"⚠️  Found {len(unparseable_rows)} rows with unparseable compositions:")
                for idx, row in unparseable_rows.head(3).iterrows():
                    print(f"   '{row['System_Name']}'")
                if len(unparseable_rows) > 3:
                    print(f"   ... and {len(unparseable_rows) - 3} more")
            
            cleaned_df = cleaned_df[parseable_mask]
        
        # 7. Reset index
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        final_size = len(cleaned_df)
        removed_count = original_size - final_size
        
        print(f"✅ Data cleaning completed:")
        print(f"   Original size: {original_size}")
        print(f"   Final size: {final_size}")
        print(f"   Removed: {removed_count} rows ({removed_count/original_size*100:.1f}%)")
        
        if final_size == 0:
            raise ValueError("No valid data remaining after cleaning!")
        
        return cleaned_df
    
    def _validate_dataframes_for_export(self):
        """
        🚀 NEW: Validate dataframes before export to ensure proper format
        """
        for name, df in [('train', self.train_df), ('valid', self.valid_df)]:
            if df.empty:
                continue
                
            # Check for missing values
            missing_system = df['System_Name'].isna().sum()
            missing_target = df['Tm (Liquidus)'].isna().sum()
            
            if missing_system > 0:
                print(f"⚠️  Warning: {missing_system} missing System_Name values in {name} set")
                df.dropna(subset=['System_Name'], inplace=True)
            
            if missing_target > 0:
                print(f"⚠️  Warning: {missing_target} missing Tm (Liquidus) values in {name} set")
                df.dropna(subset=['Tm (Liquidus)'], inplace=True)
            
            # Check for empty strings
            empty_system = (df['System_Name'].str.strip() == '').sum()
            if empty_system > 0:
                print(f"⚠️  Warning: {empty_system} empty System_Name values in {name} set")
                df = df[df['System_Name'].str.strip() != '']
            
            # Ensure target values are numeric
            try:
                df['Tm (Liquidus)'] = pd.to_numeric(df['Tm (Liquidus)'], errors='coerce')
                nan_targets = df['Tm (Liquidus)'].isna().sum()
                if nan_targets > 0:
                    print(f"⚠️  Warning: {nan_targets} non-numeric target values in {name} set")
                    df.dropna(subset=['Tm (Liquidus)'], inplace=True)
            except Exception as e:
                print(f"⚠️  Warning: Error converting targets to numeric in {name} set: {e}")
            
            # Update the dataframe
            if name == 'train':
                self.train_df = df.reset_index(drop=True)
            else:
                self.valid_df = df.reset_index(drop=True)
    
    def verify_csv_files(self, output_dir):
        """
        🚀 NEW: Verify the created CSV files are properly formatted
        
        Args:
            output_dir: Directory containing the CSV files
        """
        print("🔍 Verifying CSV file formats...")
        
        for filename in ['train.csv', 'valid.csv']:
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                continue
                
            try:
                # Read the file line by line to check format
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                print(f"📋 Checking {filename}...")
                print(f"   Total lines: {len(lines)}")
                
                # Check header
                if len(lines) > 0:
                    header = lines[0].strip()
                    expected_header = "System_Name,Tm (Liquidus)"
                    if header != expected_header:
                        print(f"   ⚠️  Unexpected header: '{header}' (expected: '{expected_header}')")
                    else:
                        print(f"   ✅ Header correct")
                
                # Check data lines
                problematic_lines = []
                for i, line in enumerate(lines[1:], 2):  # Start from line 2 (after header)
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                        
                    parts = line.split(',')
                    if len(parts) != 2:
                        problematic_lines.append((i, line, f"Expected 2 parts, got {len(parts)}"))
                    elif not parts[0].strip():
                        problematic_lines.append((i, line, "Empty composition"))
                    elif not parts[1].strip():
                        problematic_lines.append((i, line, "Empty target value"))
                    else:
                        # Try to parse target as float
                        try:
                            float(parts[1])
                        except ValueError:
                            problematic_lines.append((i, line, f"Invalid target value: '{parts[1]}'"))
                
                if problematic_lines:
                    print(f"   ⚠️  Found {len(problematic_lines)} problematic lines:")
                    for line_num, line_content, issue in problematic_lines[:3]:
                        print(f"      Line {line_num}: {issue}")
                        print(f"         Content: '{line_content}'")
                    if len(problematic_lines) > 3:
                        print(f"      ... and {len(problematic_lines) - 3} more")
                else:
                    print(f"   ✅ All data lines properly formatted")
                
                # Test reading with pandas
                test_df = pd.read_csv(filepath)
                print(f"   ✅ Successfully loaded with pandas: {len(test_df)} rows")
                
            except Exception as e:
                print(f"   ❌ Error checking {filename}: {e}")
        
        print("✅ CSV verification completed")
    
    def count_components(self, composition):
        parsed = self.parser(composition)
        parsed = [item for item in parsed if isinstance(item, tuple) and item[0] and item[1] != 0.0]
        return len(parsed)
    
    def augmenter(self, df=None):
        """Apply permutation augmentation to compositions."""
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
                        perc /= 100  # Fixed typo: was 'per /= 100'
                    if perc == 1 or perc == 100:
                        res += f"{comp}"
                    else:
                        res += f"{comp}{perc}"
                if res:  # Only append if res is not empty
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
        """
        🚀 ENHANCED: Split data into train/valid only with optional augmentation control
        
        Args:
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (train_df, valid_df)
        """
        # 🚀 NEW: Apply augmentation based on flag
        if self.apply_augmentation:
            print("🔄 Applying permutation augmentation...")
            working_df = self.augmenter(self.df)
            self.augmentation_applied = True
            print(f"   Original data: {len(self.df)} compositions")
            print(f"   Augmented data: {len(working_df)} compositions")
            print(f"   Augmentation factor: {len(working_df) / len(self.df):.1f}x")
        else:
            print("⚡ Using original data without augmentation...")
            working_df = self.df.copy()
            self.augmentation_applied = False
            print(f"   Data size: {len(working_df)} compositions")
        
        if working_df.empty:
            raise ValueError("No data available for splitting. Check your input data and augmentation settings.")
        
        # Add component count as a column for stratification
        working_df['n_components'] = working_df['System_Name'].apply(self.count_components)
        
        # Split into train and validation sets
        X = working_df[['System_Name', 'n_components']]
        y = working_df['Tm (Liquidus)']
        
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, 
            test_size=self.valid_ratio/100,
            stratify=X['n_components'],
            random_state=random_state
        )
        
        # Reconstruct dataframes
        self.train_df = pd.DataFrame({
            'System_Name': X_train['System_Name'],
            'Tm (Liquidus)': y_train
        }).reset_index(drop=True)
        
        self.valid_df = pd.DataFrame({
            'System_Name': X_valid['System_Name'],
            'Tm (Liquidus)': y_valid
        }).reset_index(drop=True)
        
        return self.train_df, self.valid_df
    
    def stratified_split_composition_based(self, random_state=42):
        """
        🚀 NEW: Alternative splitting method that avoids composition leakage
        
        This method ensures that no composition appears in multiple splits,
        which is crucial for extrapolation evaluation.
        
        Args:
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (train_df, valid_df)
        """
        print("🎯 Using composition-based splitting (no leakage between splits)...")
        
        # Work with original compositions first
        original_df = self.df.copy()
        original_df['n_components'] = original_df['System_Name'].apply(self.count_components)
        
        # Split original compositions
        X = original_df[['System_Name', 'n_components']]
        y = original_df['Tm (Liquidus)']
        
        # Split into train and validation compositions
        X_train_orig, X_valid_orig, y_train_orig, y_valid_orig = train_test_split(
            X, y,
            test_size=self.valid_ratio/100,
            stratify=X['n_components'],
            random_state=random_state
        )
        
        # Now apply augmentation to each split separately if enabled
        if self.apply_augmentation:
            print("🔄 Applying augmentation to each split separately...")
            
            # Create temporary dataframes for each split
            train_orig_df = pd.DataFrame({
                'System_Name': X_train_orig['System_Name'],
                'Tm (Liquidus)': y_train_orig
            })
            valid_orig_df = pd.DataFrame({
                'System_Name': X_valid_orig['System_Name'],
                'Tm (Liquidus)': y_valid_orig
            })
            
            # Augment each split
            self.train_df = self.augmenter(train_orig_df)
            self.valid_df = self.augmenter(valid_orig_df)
            
            self.augmentation_applied = True
            
            print(f"   Train: {len(train_orig_df)} → {len(self.train_df)} compositions")
            print(f"   Valid: {len(valid_orig_df)} → {len(self.valid_df)} compositions")
            
        else:
            # Use original compositions without augmentation
            self.train_df = pd.DataFrame({
                'System_Name': X_train_orig['System_Name'],
                'Tm (Liquidus)': y_train_orig
            }).reset_index(drop=True)
            
            self.valid_df = pd.DataFrame({
                'System_Name': X_valid_orig['System_Name'],
                'Tm (Liquidus)': y_valid_orig
            }).reset_index(drop=True)
            
            self.augmentation_applied = False
            
            print(f"   Train: {len(self.train_df)} compositions")
            print(f"   Valid: {len(self.valid_df)} compositions")
        
        return self.train_df, self.valid_df
    
    def validate_split(self, verbose=True):
        """Validate that the split maintains desired ratios"""
        total = len(self.train_df) + len(self.valid_df)
        
        if total == 0:
            print("No data in splits. Run stratified_split_by_components first.")
            return None
            
        actual_train_ratio = len(self.train_df) / total * 100
        actual_valid_ratio = len(self.valid_df) / total * 100
        
        validation_results = {
            'target_ratios': {
                'train': self.train_ratio,
                'valid': self.valid_ratio
            },
            'actual_ratios': {
                'train': actual_train_ratio,
                'valid': actual_valid_ratio
            },
            'differences': {
                'train': abs(actual_train_ratio - self.train_ratio),
                'valid': abs(actual_valid_ratio - self.valid_ratio)
            },
            'augmentation_applied': self.augmentation_applied
        }
        
        if verbose:
            print(f"🎯 Augmentation status: {'Applied' if self.augmentation_applied else 'Not applied'}")
            print(f"Target ratios: Train={self.train_ratio}%, Valid={self.valid_ratio}%")
            print(f"Actual ratios: Train={actual_train_ratio:.2f}%, Valid={actual_valid_ratio:.2f}%")
            print(f"Differences:   Train={validation_results['differences']['train']:.2f}%, "
                  f"Valid={validation_results['differences']['valid']:.2f}%")
            
            # Check if within tolerance (e.g., 2%)
            tolerance = 2.0
            if any(diff > tolerance for diff in validation_results['differences'].values()):
                print("⚠️  WARNING: Some actual ratios deviate from target by more than 2%")
            else:
                print("✓ All ratios are within 2% tolerance")
        
        return validation_results
    
    def plot_distribution_validation(self, save_path=None):
        """Create visualization to verify distribution matching"""
        # Validate split first
        validation_results = self.validate_split(verbose=False)
        if validation_results is None:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Data Split Distribution Validation (Train/Valid Only)\n(Augmentation: {"Applied" if self.augmentation_applied else "Not Applied"})', fontsize=16)
        
        # 1. Target vs Actual Ratios Bar Chart
        ax1 = axes[0, 0]
        x = np.arange(2)
        width = 0.35
        
        target_values = [validation_results['target_ratios']['train'],
                        validation_results['target_ratios']['valid']]
        actual_values = [validation_results['actual_ratios']['train'],
                        validation_results['actual_ratios']['valid']]
        
        bars1 = ax1.bar(x - width/2, target_values, width, label='Target', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, actual_values, width, label='Actual', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Target vs Actual Split Ratios')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Train', 'Valid'])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        # 2. Difference from Target
        ax2 = axes[0, 1]
        differences = list(validation_results['differences'].values())
        colors = ['green' if d <= 2.0 else 'orange' if d <= 5.0 else 'red' for d in differences]
        bars = ax2.bar(['Train', 'Valid'], differences, color=colors, alpha=0.7)
        
        ax2.axhline(y=2.0, color='green', linestyle='--', label='2% tolerance')
        ax2.axhline(y=5.0, color='orange', linestyle='--', label='5% tolerance')
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Absolute Difference (%)')
        ax2.set_title('Deviation from Target Ratios')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, diff in zip(bars, differences):
            ax2.annotate(f'{diff:.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        # 3. Component Distribution Stacked Bar Chart
        ax3 = axes[1, 0]
        
        # Calculate component distributions
        train_comp = self.categorize_by_component_count(self.train_df)
        valid_comp = self.categorize_by_component_count(self.valid_df)
        
        # Get all component numbers
        all_components = sorted(set(list(train_comp.keys()) + list(valid_comp.keys())))
        
        # Prepare data for stacked bar chart
        train_counts = [len(train_comp.get(n, [])) for n in all_components]
        valid_counts = [len(valid_comp.get(n, [])) for n in all_components]
        
        # Create stacked bar chart
        x = np.arange(len(all_components))
        p1 = ax3.bar(x, train_counts, label='Train', alpha=0.8)
        p2 = ax3.bar(x, valid_counts, bottom=train_counts, label='Valid', alpha=0.8)
        
        ax3.set_xlabel('Number of Components')
        ax3.set_ylabel('Count')
        ax3.set_title('Component Distribution Across Splits')
        ax3.set_xticks(x)
        ax3.set_xticklabels(all_components)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Pie Chart showing overall distribution
        ax4 = axes[1, 1]
        sizes = [len(self.train_df), len(self.valid_df)]
        labels = [f'Train\n{len(self.train_df)} samples\n({validation_results["actual_ratios"]["train"]:.1f}%)',
                 f'Valid\n{len(self.valid_df)} samples\n({validation_results["actual_ratios"]["valid"]:.1f}%)']
        colors = ['#66b3ff', '#99ff99']
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='',
                                           startangle=90, textprops={'fontsize': 10})
        ax4.set_title('Overall Data Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution validation plot saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_component_distribution_details(self, save_path=None):
        """Create detailed visualization of component distribution"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calculate component distributions
        train_comp = self.categorize_by_component_count(self.train_df)
        valid_comp = self.categorize_by_component_count(self.valid_df)
        
        # Get all component numbers
        all_components = sorted(set(list(train_comp.keys()) + list(valid_comp.keys())))
        
        # Prepare data
        x = np.arange(len(all_components))
        width = 0.35
        
        train_counts = [len(train_comp.get(n, [])) for n in all_components]
        valid_counts = [len(valid_comp.get(n, [])) for n in all_components]
        
        # Create grouped bar chart
        bars1 = ax.bar(x - width/2, train_counts, width, label='Train', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x + width/2, valid_counts, width, label='Valid', alpha=0.8, color='#ff7f0e')
        
        # Customize chart
        ax.set_xlabel('Number of Components', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        title = f'Distribution of Component Types Across Train/Valid Sets\n(Augmentation: {"Applied" if self.augmentation_applied else "Not Applied"})'
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        
        # Create custom labels
        labels = []
        for n in all_components:
            if n == 1:
                labels.append('1\n(Single)')
            elif n == 2:
                labels.append('2\n(Double)')
            elif n == 3:
                labels.append('3\n(Triple)')
            elif n == 4:
                labels.append('4\n(Quadruple)')
            else:
                labels.append(str(n))
        
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=9)
        
        # Add percentage distribution text
        total_train = sum(train_counts)
        total_valid = sum(valid_counts)
        
        # Create text box with percentages
        textstr = 'Component Distribution Percentages:\n'
        for i, n in enumerate(all_components):
            comp_label = ['Single', 'Double', 'Triple', 'Quadruple'][n-1] if n <= 4 else f'{n}-component'
            if total_train > 0:
                train_pct = train_counts[i] / total_train * 100
            else:
                train_pct = 0
            if total_valid > 0:
                valid_pct = valid_counts[i] / total_valid * 100
            else:
                valid_pct = 0
            
            textstr += f'{comp_label}: Train={train_pct:.1f}%, Valid={valid_pct:.1f}%\n'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.98, 0.97, textstr.strip(), transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Component distribution plot saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def get_split_info(self, include_component_distribution=True):
        total_size = len(self.train_df) + len(self.valid_df)
        
        info = {
            'train_size': len(self.train_df),
            'valid_size': len(self.valid_df),
            'total_size': total_size,
            'augmentation_applied': self.augmentation_applied
        }
        
        if total_size > 0:
            info['train_percentage'] = (len(self.train_df) / total_size) * 100
            info['valid_percentage'] = (len(self.valid_df) / total_size) * 100
        
        if include_component_distribution:
            train_components = self.categorize_by_component_count(self.train_df)
            valid_components = self.categorize_by_component_count(self.valid_df)
            
            original_components = self.categorize_by_component_count(self.df)
            
            info['component_distribution'] = {
                'train': {k: len(v) for k, v in train_components.items()},
                'valid': {k: len(v) for k, v in valid_components.items()}
            }
            
            info['original_composition_counts'] = {
                k: len(v) for k, v in original_components.items()
            }
            
            if self.augmentation_applied:
                augmented_df = self.augmenter()
                augmented_components = self.categorize_by_component_count(augmented_df)
                info['augmented_composition_counts'] = {
                    k: len(v) for k, v in augmented_components.items()
                }
        
        return info
    
    def save_description(self, output_dir, file_name="description.txt"):
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, file_name)
        
        split_info = self.get_split_info(include_component_distribution=True)
        validation_results = self.validate_split(verbose=False)
        
        original_components = self.categorize_by_component_count(self.df)
        
        with open(file_path, 'w') as f:
            f.write("=================================================\n")
            f.write("     DATASET DESCRIPTION & SPLITTING (TRAIN/VALID)  \n")
            f.write("=================================================\n\n")
            
            f.write("CONFIGURATION\n")
            f.write("-------------\n")
            f.write(f"Augmentation applied: {self.augmentation_applied}\n")
            f.write(f"Augmentation setting: {self.apply_augmentation}\n")
            f.write(f"Split ratio: Train {self.train_ratio}% / Valid {self.valid_ratio}%\n\n")
            
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
            
            if self.augmentation_applied:
                augmented_df = self.augmenter()
                augmented_components = self.categorize_by_component_count(augmented_df)
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
            else:
                f.write("\nAUGMENTATION INFORMATION\n")
                f.write("-----------------------\n")
                f.write("No augmentation was applied.\n\n")
            
            f.write("\nSPLIT INFORMATION\n")
            f.write("----------------\n")
            f.write(f"Train size: {split_info['train_size']} compositions ({split_info['train_percentage']:.2f}%)\n")
            f.write(f"Validation size: {split_info['valid_size']} compositions ({split_info['valid_percentage']:.2f}%)\n")
            f.write(f"Total size: {split_info['total_size']} compositions\n\n")
            
            # Add validation results
            if validation_results:
                f.write("Split Validation Results:\n")
                f.write(f"Target ratios: Train={validation_results['target_ratios']['train']}%, "
                       f"Valid={validation_results['target_ratios']['valid']}%\n")
                f.write(f"Actual ratios: Train={validation_results['actual_ratios']['train']:.2f}%, "
                       f"Valid={validation_results['actual_ratios']['valid']:.2f}%\n")
                f.write(f"Differences:   Train={validation_results['differences']['train']:.2f}%, "
                       f"Valid={validation_results['differences']['valid']:.2f}%\n\n")
            
            f.write("Component distribution after splitting:\n")
            component_dist = split_info['component_distribution']
            
            f.write("\nNumber of compositions by component count:\n")
            max_components = max(
                max(component_dist['train'].keys(), default=0),
                max(component_dist['valid'].keys(), default=0)
            )
            
            f.write(f"{'Components':<12} {'Label':<10} {'Train':<10} {'Valid':<10}\n")
            f.write("-" * 50 + "\n")
            
            for n in range(1, max_components + 1):
                label = "single" if n == 1 else "double" if n == 2 else "triple" if n == 3 else "quadruple"
                train_count = component_dist['train'].get(n, 0)
                valid_count = component_dist['valid'].get(n, 0)
                f.write(f"{n:<12} {label:<10} {train_count:<10} {valid_count:<10}\n")
            
            f.write("\n\nDATA SPLITTING METHODOLOGY\n")
            f.write("-------------------------\n")
            if self.augmentation_applied:
                f.write("The data was split using sklearn's stratified train_test_split with augmentation:\n")
                f.write("1. Data augmentation was performed by generating all possible permutations of\n")
                f.write("   the chemical compositions to increase the dataset size and diversity.\n")
                f.write("2. All compositions (including single-element) are distributed across train/valid sets.\n")
                f.write("3. Stratification by component count ensures equal proportional representation in each split.\n")
                f.write("4. The split maintains the target 85/15 ratio as closely as possible.\n\n")
            else:
                f.write("The data was split using sklearn's stratified train_test_split without augmentation:\n")
                f.write("1. Original compositions were used without permutation augmentation.\n")
                f.write("2. All compositions (including single-element) are distributed across train/valid sets.\n")
                f.write("3. Stratification by component count ensures equal proportional representation in each split.\n")
                f.write("4. The split maintains the target 85/15 ratio as closely as possible.\n")
                f.write("5. This approach avoids potential data leakage that could occur with augmentation.\n\n")
        
        print(f"Detailed description saved to: {file_path}")
        return file_path

    def __call__(self, 
                output_dir, 
                random_state, 
                save, 
                verbose,
                create_plots=True,
                split_method='standard'):
        """
        🚀 ENHANCED: Execute the complete pipeline with configurable augmentation and splitting
        
        Args:
            output_dir: Directory to save results
            random_state: Random seed for reproducibility
            save: Whether to save results to files
            verbose: Whether to print detailed information
            create_plots: Whether to create visualization plots
            split_method: 'standard' or 'composition_based' (for extrapolation)
        """
        if verbose:
            print("🚀 Starting data splitting pipeline (Train/Valid only)...")
            print(f"   Augmentation: {'Enabled' if self.apply_augmentation else 'Disabled'}")
            print(f"   Split method: {split_method}")
            print(f"   Original data size: {len(self.df)} rows")
            print(f"   Target split: Train {self.train_ratio}% / Valid {self.valid_ratio}%")
            
            original_components = self.categorize_by_component_count(self.df)
            print("\nOriginal composition counts by component number:")
            for n_comp, indices in sorted(original_components.items()):
                label = "single" if n_comp == 1 else "double" if n_comp == 2 else "triple" if n_comp == 3 else "quadruple"
                print(f"{n_comp}({label}) - number of compositions found: {len(indices)}")
                if indices:
                    print("First 2 sample compositions:")
                    for i, idx in enumerate(indices[:2]):
                        print(f"  {i+1}. {self.df.iloc[idx]['System_Name']} - {self.df.iloc[idx]['Tm (Liquidus)']}")
        
        # Perform the split based on method
        if split_method == 'composition_based':
            train_df, valid_df = self.stratified_split_composition_based(random_state=random_state)
        else:
            train_df, valid_df = self.stratified_split_by_components(random_state=random_state)
        
        if verbose:
            print("\nSplit information:")
            split_info = self.get_split_info(include_component_distribution=True)
            
            for key, value in split_info.items():
                if key not in ['component_distribution', 'original_composition_counts', 'augmented_composition_counts']:
                    if 'percentage' in key:
                        print(f"{key}: {value:.2f}%")
                    elif key == 'augmentation_applied':
                        print(f"{key}: {value}")
                    else:
                        print(f"{key}: {value}")
            
            # Validate the split
            print("\nValidating split ratios...")
            self.validate_split()
            
            print("\nComponent distribution after splitting:")
            component_dist = split_info['component_distribution']
            
            print("\nNumber of compositions by component count:")
            max_components = max(
                max(component_dist['train'].keys(), default=0),
                max(component_dist['valid'].keys(), default=0)
            )
            
            print(f"{'Components':<12} {'Label':<10} {'Train':<10} {'Valid':<10}")
            print("-" * 50)
            
            for n in range(1, max_components + 1):
                label = "single" if n == 1 else "double" if n == 2 else "triple" if n == 3 else "quadruple"
                train_count = component_dist['train'].get(n, 0)
                valid_count = component_dist['valid'].get(n, 0)
                print(f"{n:<12} {label:<10} {train_count:<10} {valid_count:<10}")
        
        # Create visualizations
        if create_plots:
            os.makedirs(output_dir, exist_ok=True)
            print("\nCreating distribution validation plots...")
            self.plot_distribution_validation(save_path=os.path.join(output_dir, 'distribution_validation.png'))
            self.plot_component_distribution_details(save_path=os.path.join(output_dir, 'component_distribution.png'))
        
        if save:
            os.makedirs(output_dir, exist_ok=True)
            
            # 🚀 NEW: Validate data before saving
            print("🔍 Validating data before saving...")
            self._validate_dataframes_for_export()
            
            self.train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
            self.valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
            
            self.save_description(output_dir)
            
            # Verify the created CSV files
            self.verify_csv_files(output_dir)
            
            if verbose:
                print(f"\nData saved to: {output_dir}")
                print(f"✅ Train CSV: {len(self.train_df)} rows")
                print(f"✅ Valid CSV: {len(self.valid_df)} rows")
        
        if verbose:
            print("\n🎯 Pipeline completed successfully!")
        
        return self.train_df, self.valid_df


# Helper functions for easy usage
def create_splitter_for_standard_training(data):
    """Create a DataSplitter configured for standard training with augmentation"""
    return DataSpliter(data, apply_augmentation=True)

def create_splitter_for_extrapolation(data):
    """Create a DataSplitter configured for extrapolation training without augmentation"""
    return DataSpliter(data, apply_augmentation=False)

def clean_csv_file(input_path, output_path=None):
    """
    🚀 NEW: Clean a problematic CSV file
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output cleaned CSV file (if None, overwrites input)
    
    Returns:
        str: Path to cleaned file
    """
    if output_path is None:
        output_path = input_path
    
    print(f"🧹 Cleaning CSV file: {input_path}")
    
    cleaned_lines = []
    problematic_lines = []
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # Keep header
    if lines:
        header = lines[0].strip()
        cleaned_lines.append(header)
        
        for i, line in enumerate(lines[1:], 2):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(',')
            if len(parts) != 2:
                problematic_lines.append((i, line, f"Wrong number of parts: {len(parts)}"))
                continue
                
            composition = parts[0].strip()
            target_str = parts[1].strip()
            
            if not composition:
                problematic_lines.append((i, line, "Empty composition"))
                continue
                
            if not target_str:
                problematic_lines.append((i, line, "Empty target"))
                continue
                
            # Try to parse target as float
            try:
                target_val = float(target_str)
                cleaned_lines.append(f"{composition},{target_val}")
            except ValueError:
                problematic_lines.append((i, line, f"Invalid target: '{target_str}'"))
                continue
    
    # Write cleaned file
    with open(output_path, 'w') as f:
        for line in cleaned_lines:
            f.write(line + '\n')
    
    print(f"✅ Cleaned file saved to: {output_path}")
    print(f"   Original lines: {len(lines)}")
    print(f"   Cleaned lines: {len(cleaned_lines)}")
    print(f"   Removed: {len(lines) - len(cleaned_lines)} lines")
    
    if problematic_lines:
        print(f"   Problems found in {len(problematic_lines)} lines:")
        for line_num, line_content, issue in problematic_lines[:3]:
            print(f"      Line {line_num}: {issue}")
        if len(problematic_lines) > 3:
            print(f"      ... and {len(problematic_lines) - 3} more")
    
    return output_path


if __name__ == "__main__":
    # Example usage - load your data
    example_df = pd.read_csv('Data/All_Data_Based_on_05_06_2025/outlayer_cleaned-06-06-2025-data.csv')

    # 🚀 EXAMPLE 1: Standard training with augmentation (85% train, 15% valid)
    print("=" * 60)
    print("EXAMPLE 1: Standard Training (with augmentation)")
    print("=" * 60)
    try:
        splitter1 = create_splitter_for_standard_training(example_df)
        
        train1, valid1 = splitter1(
            output_dir='./Data/Standard_Training_Split',
            random_state=123,
            save=True,
            verbose=True,
            create_plots=True
        )
        print("✅ Standard training split completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in standard training split: {e}")
    
    # 🚀 EXAMPLE 2: Extrapolation training without augmentation (85% train, 15% valid)
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Extrapolation Training (no augmentation)")
    print("=" * 60)
    try:
        splitter2 = create_splitter_for_extrapolation(example_df)
        
        train2, valid2 = splitter2(
            output_dir='./Data/Extrapolation_Training_Split',
            random_state=123,
            save=True,
            verbose=True,
            create_plots=True,
            split_method='composition_based'  # Use composition-based splitting
        )
        print("✅ Extrapolation training split completed successfully!")
        
        # Compare the results
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"Standard training dataset size: {len(train1)} train, {len(valid1)} valid")
        print(f"Extrapolation training dataset size: {len(train2)} train, {len(valid2)} valid")
        print(f"Augmentation factor: {len(train1) / len(train2):.1f}x")
        
    except Exception as e:
        print(f"❌ Error in extrapolation training split: {e}")
    
    # # 🚀 EXAMPLE 3: Cleaning problematic CSV files
    # print("\n" + "=" * 60)
    # print("EXAMPLE 3: Cleaning Problematic CSV Files")
    # print("=" * 60)
    
    # # Example of how to clean existing CSV files with problems
    # problematic_files = [
    #     './Data/Standard_Training_Split/train.csv',
    #     './Data/Standard_Training_Split/valid.csv',
    #     './Data/Extrapolation_Training_Split/train.csv',
    #     './Data/Extrapolation_Training_Split/valid.csv'
    # ]
    
    # for file_path in problematic_files:
    #     if os.path.exists(file_path):
    #         try:
    #             clean_csv_file(file_path)
    #         except Exception as e:
    #             print(f"❌ Error cleaning {file_path}: {e}")
    #     else:
    #         print(f"⏭️  Skipping {file_path} (file not found)")
    
    print("\n🎯 All examples completed!")
    print("\n💡 Tips to avoid data format issues:")
    print("   1. Always use the cleaned DataSplitter (it validates data automatically)")
    print("   2. Check your original CSV file for missing values or malformed rows")
    print("   3. Use clean_csv_file() function to fix existing problematic CSV files")
    print("   4. Verify CSV files with verify_csv_files() before training")