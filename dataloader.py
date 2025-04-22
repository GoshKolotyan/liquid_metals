import torch
from tokenizer import LM_Tokenizer 
from torch.utils.data import Dataset, DataLoader

class LM_Dataset(Dataset):
    def __init__(self, data_path, tokenizer_class):
        self.data = []
        self.tokenizer = tokenizer_class()  # Initialize the tokenizer class
        
        with open(data_path, 'r') as f:
            lines = f.readlines()
            # Skip header if it exists
            for line in lines[1:]:
                composition, target = line.strip().split(',')
                # print("Composition:", type(composition), "Target:", type(target))
                # print("Composition:", composition, "Target:", float(target))
                self.data.append((composition, float(target)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        composition, target = self.data[idx]
        composition_tensor, target_tensor = self.tokenizer.encode(composition, target)
        return composition_tensor, target_tensor

DS = LM_Dataset('./Data/cleaned_data.csv', LM_Tokenizer)
print("Dataset size:", len(DS))
print("First item in dataset:", DS[0])