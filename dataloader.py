import torch
from tokenizer import LM_Tokenizer 
from torch.utils.data import Dataset, DataLoader

class LM_Dataset(Dataset):
    def __init__(self, data_path, tokenizer_class):
        self.data = []
        self.tokenizer = tokenizer_class()  
        
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
def collate_fn(batch):
    """Custom collate function to handle variable sequence lengths"""
    # Separate tokens and targets
    tokens = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Get max sequence length in this batch
    max_seq_len = max(token.size(0) for token in tokens)
    feature_dim = tokens[0].size(1)
    
    # Create padded batch
    padded_tokens = torch.zeros(len(batch), max_seq_len, feature_dim)
    for i, token in enumerate(tokens):
        padded_tokens[i, :token.size(0), :] = token
    
    # Stack targets
    targets = torch.stack(targets)
    
    return padded_tokens, targets

DS = LM_Dataset('./Data/cleaned_data.csv', LM_Tokenizer)


batch_size = 2
# Create DataLoader with custom collate_fn
train_loader = DataLoader(
    DS,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

for batch in train_loader:
    composition_tensor, target_tensor = batch
    print("Batch composition tensor shape:", composition_tensor.shape)
    print("Batch target tensor shape:", target_tensor.shape)
    print("Batch composition tensor:\n", composition_tensor)
    print("Batch target tensor:\n", target_tensor)
