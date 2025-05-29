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
    """Custom collate function to handle variable sequence lengths"""
    # Check if batch contains targets or just compositions
    if isinstance(batch[0], tuple):
        # Training/validation data with targets
        tokens = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        max_seq_len = max(token.size(0) for token in tokens)
        feature_dim = tokens[0].size(1)
        
        padded_tokens = torch.zeros(len(batch), max_seq_len, feature_dim)
        for i, token in enumerate(tokens):
            padded_tokens[i, :token.size(0), :] = token
        
        targets = torch.stack(targets)
        
        return padded_tokens, targets
    else:
        # Inference data without targets
        tokens = batch
        
        max_seq_len = max(token.size(0) for token in tokens)
        feature_dim = tokens[0].size(1)
        
        padded_tokens = torch.zeros(len(batch), max_seq_len, feature_dim)
        for i, token in enumerate(tokens):
            padded_tokens[i, :token.size(0), :] = token
        
        return padded_tokens

# train = LM_Dataset(data_path="Data/generated_compositions.csv")
# # valid = LM_Dataset(data_path="Data/Component_Stratified_Split/valid.csv")
# # print(sample_data)
# # print(sample_data[0].shape[1])

# train_loader = DataLoader(dataset=train, batch_size=64)
# # valid_load = DataLoader(dataset=valid, batch_size=2048)


# for batch in train_loader:
#     composition_tensor, target_tensor = batch
#     print("Batch composition tensor shape:", composition_tensor.shape)
#     print("Batch target tensor shape:", target_tensor.shape)
#     # print("Batch composition tensor:\n", composition_tensor)
#     # print("Batch target tensor:\n", batch)
#     # break




# # example 
# Composition: Sn0.133Bi0.5Cd0.1Pb0.267 Target: 72.14
# (tensor([[encoder_number of 1 element, percentage of 1 element, atomic radius of 1 element, electronegativity of 1 elemnt, metling point of 1],
#         [encoder_number of 2 element, percentage of 2 element, atomic radius of 2 element, electronegativity of 2 elemnt, metling point of 2],
#         [encoder_number of 3 element, percentage of 3 element, atomic radius of 3 element, electronegativity of 3 elemnt, metling point of 3],
#         [encoder_number of 4 element, percentage of 4 element, atomic radius of 4 element, electronegativity of 4 elemnt, metling point of 4],
#         [entropy_of_mixing, mix_enthalpy(0), electronegativity_difference, atomic_radius_difference, melting_point_difference]]), 

#         [melting point of composition]))