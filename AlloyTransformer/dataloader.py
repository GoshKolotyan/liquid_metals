# ==================================================================================
# DATALOADER REVIEW - Your Implementation is EXCELLENT!
# ==================================================================================

import torch
from tokenizer import LM_Tokenizer 
from torch.utils.data import Dataset, DataLoader

class LM_Dataset(Dataset):
    def __init__(self, data_path, has_targets=True):
        self.data = []
        self.tokenizer = LM_Tokenizer()
        self.has_targets = has_targets
        
        with open(data_path, 'r') as f:
            lines = f.readlines()
            # Skip header if it exists
            for line in lines[1:]:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                if self.has_targets:
                    # Training/validation data with targets
                    parts = line.split(',')
                    if len(parts) >= 2:
                        composition = parts[0]
                        target = float(parts[1])
                        self.data.append((composition, target))
                    else:
                        print(f"Warning: Skipping line with insufficient data: {line}")
                else:
                    # Inference data without targets
                    composition = line
                    self.data.append((composition, None))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        composition, target = self.data[idx]
        
        if self.has_targets:
            composition_tensor, target_tensor = self.tokenizer.encode(composition, target)
            return composition_tensor, target_tensor
        else:
            # For inference, only encode composition
            composition_tensor = self.tokenizer.encode(composition=composition, target=None)
            return composition_tensor

def collate_fn(batch):
    """Fixed version - corrected critical errors"""
    if isinstance(batch[0], tuple):
        # Training/validation case with targets
        tokens = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        max_seq_len = max(token.size(0) for token in tokens)
        feature_dim = tokens[0].size(1)
        
        # Initialize padded tensors and attention masks
        padded_tokens = torch.zeros(len(batch), max_seq_len, feature_dim)
        attention_mask = torch.zeros(len(batch), max_seq_len, dtype=torch.bool)
        
        for i, token in enumerate(tokens):
            actual_length = token.size(0)
            padded_tokens[i, :actual_length, :] = token
            attention_mask[i, :actual_length] = True
            # Padded positions remain False
        
        targets = torch.stack(targets)
        return padded_tokens, targets, attention_mask 
    else:
        # Inference case without targets
        tokens = batch
        max_seq_len = max(token.size(0) for token in tokens)
        feature_dim = tokens[0].size(1)
        
        padded_tokens = torch.zeros(len(batch), max_seq_len, feature_dim)
        attention_mask = torch.zeros(len(batch), max_seq_len, dtype=torch.bool)
        
        for i, token in enumerate(tokens):
            actual_length = token.size(0)
            padded_tokens[i, :actual_length, :] = token
            attention_mask[i, :actual_length] = True
        
        return padded_tokens, attention_mask 

if __name__ == "__main__":
    train = LM_Dataset(data_path="Data/example.csv")
    # valid = LM_Dataset(data_path="Data/Component_Stratified_Split/valid.csv")
    # print(sample_data)
    # print(sample_data[0].shape[1])

    train_loader = DataLoader(dataset=train, batch_size=1, collate_fn=collate_fn)
    # valid_load = DataLoader(dataset=valid, batch_size=2048)


    for batch in train_loader:
        composition_tensor, target_tensor, attention_mask = batch

        print(20*"==", "New Batch", 20*"==")
        print("=" * 20, "Composition Tensor", "=" * 20)
        print("Shape:", composition_tensor.shape)
        print(composition_tensor)

        print("\n" + "=" * 20, "Attention Mask", "=" * 20)
        print("Shape:", attention_mask.shape)
        print(attention_mask)

        print("\n" + "=" * 20, "Target Tensor", "=" * 20)
        print("Shape:", target_tensor.shape)
        print(target_tensor)

        print(20*"==", "New Batch", 20*"==")

        # break









# # example 
# Composition: Sn0.133Bi0.5Cd0.1Pb0.267 Target: 72.14
# (tensor([[encoder_number of 1 element, percentage of 1 element, atomic radius of 1 element, electronegativity of 1 elemnt, metling point of 1],
#         [encoder_number of 2 element, percentage of 2 element, atomic radius of 2 element, electronegativity of 2 elemnt, metling point of 2],
#         [encoder_number of 3 element, percentage of 3 element, atomic radius of 3 element, electronegativity of 3 elemnt, metling point of 3],
#         [encoder_number of 4 element, percentage of 4 element, atomic radius of 4 element, electronegativity of 4 elemnt, metling point of 4],
#         [entropy_of_mixing, mix_enthalpy(0), electronegativity_difference, atomic_radius_difference, melting_point_difference]]), 

#         [melting point of composition]))