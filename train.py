import torch
import torch.nn as nn
import torch.optim as optim

from tokenizer import LM_Tokenizer
from dataloader import LM_Dataset
from model import ChemicalTransformer

def train_model(config):
    # Set random seed
    torch.manual_seed(config.get('seed', 42))
    
    
    # Load all data at once
    dataset = LM_Dataset(config['data_path'])
    
    # Get all compositions and targets
    all_compositions = []
    all_targets = []
    for i in range(len(dataset)):
        composition, target = dataset[i]
        all_compositions.append(composition)
        all_targets.append(target)
    
    # Convert targets to tensor 
    targets = torch.tensor([t.item() if isinstance(t, torch.Tensor) else t for t in all_targets], dtype=torch.float)
    
    # Initialize model with sample data
    # Use the first composition directly as it's already a tensor
    sample_tokens = all_compositions[0]
    feature_dim = sample_tokens.shape[1]
    max_seq_len = max(comp.shape[0] for comp in all_compositions)
    
    # Create padded input tensor
    inputs = torch.zeros(len(all_compositions), max_seq_len, feature_dim)
    for i, comp_tensor in enumerate(all_compositions):
        inputs[i, :comp_tensor.shape[0], :] = comp_tensor
    
    # Initialize model
    model = ChemicalTransformer(
        feature_dim=feature_dim,
        hidden_dim=config.get('hidden_dim', 64),
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.1),
        max_seq_length=max_seq_len
    )
    
    # Setup device, optimizer and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    criterion = nn.L1Loss()
    
    # Training Loop
    print(f"Training for {config.get('num_epochs', 25)} epochs:")
    
    for epoch in range(config.get('num_epochs', 25)):
        model.train()
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(inputs)
        
        # Compute loss
        loss = criterion(predictions.squeeze(), targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{config.get('num_epochs', 25)}, Loss: {loss.item():.4f}")
    
    print("Training completed!")
    return model

# After training, call this function with a sample input
def main():
    # Configuration
    config = {
        'data_path': './Data/generated_compositions.csv',
        'seed': 42,
        'learning_rate': 0.001,
        'num_epochs': 1500,
        'hidden_dim': 64,
        'num_heads': 16,
        'num_layers': 4,
        'dropout': 0.1
    }
    
    # Train the model
    trained_model = train_model(config)

if __name__ == '__main__':
    main()