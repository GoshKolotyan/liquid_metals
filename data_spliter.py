import os 
import pandas as pd
from itertools import permutations
from helper import parse_compostion
from sklearn.model_selection import train_test_split

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
        self.train_ratio = 70  # percentage
        self.valid_ratio = 15  # percentage
        self.test_ratio = 15   # percentage
        self.parser = parse_compostion
    
    def augmenter(self, df=None):
        """Augment data by creating all possible permutations of chemical compositions"""
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
                    res += f"{comp}" if perc == 1 else f"{comp}{perc}"
                    
                gens.append({'System_Name': res, 'Tm (Liquidus)': target})

        return pd.DataFrame(gens)
    
    def split_augmented_data(self, random_state=42):
        """Split augmented data into train/valid/test sets"""
        augmented_df = self.augmenter(self.df)
        
        if augmented_df.empty:
            raise ValueError("No augmented data generated. Check your input data.")
        
        train_ratio_decimal = self.train_ratio / 100
        valid_ratio_decimal = self.valid_ratio / 100
        test_ratio_decimal = self.test_ratio / 100 
        
        train_valid_df, self.test_df = train_test_split(
            augmented_df, 
            test_size=test_ratio_decimal,
            random_state=random_state,
            shuffle=True
        )
        
        valid_size_relative = valid_ratio_decimal / (train_ratio_decimal + valid_ratio_decimal)
        
        self.train_df, self.valid_df = train_test_split(
            train_valid_df,
            test_size=valid_size_relative,
            random_state=random_state,
            shuffle=True
        )
        
        return self.train_df, self.valid_df, self.test_df
    
    def get_split_info(self):
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
        
        return info
    
    def save_data(self, output_dir='./Data/Merged_Splited'):
        os.makedirs(output_dir, exist_ok=True)
        self.train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        self.valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
        self.test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        print(f"Data saved to {output_dir}")

    def __call__(self, 
                 output_dir='./Data/Merged_Splited', 
                 random_state=42, 
                 save=True, 
                 verbose=True):

        if verbose:
            print("Starting data augmentation and splitting pipeline...")
            print(f"Original data size: {len(self.df)} rows")
        
        train_df, valid_df, test_df = self.split_augmented_data(random_state=random_state)
        
        if verbose:
            print("\nSplit information:")
            split_info = self.get_split_info()
            for key, value in split_info.items():
                if 'percentage' in key:
                    print(f"{key}: {value:.2f}%")
                else:
                    print(f"{key}: {value}")
        
        if save:
            self.save_data(output_dir)
            if verbose:
                print(f"\nData saved to: {output_dir}")
        
        if verbose:
            print("\nPipeline completed successfully!")
        
        return train_df, valid_df, test_df


if __name__ == "__main__":
    example_df = pd.read_csv('./Data/Merged_All.csv')
    
    splitter = DataSpliter(example_df)
    
    train, valid, test = splitter()
    
    train, valid, test = splitter(
        output_dir='./Data/Custom_Output',
        random_state=123,
        save=True,
        verbose=True
    )
    
    augmented = splitter.augmenter()
    print(f"Total augmented rows: {len(augmented)}")