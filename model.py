import torch
import torch.nn as nn
import torch.nn.functional as F

class ChemicalTransformer(nn.Module):
    def __init__(self, 
                 feature_dim=5,          # Dimension of each token (element or calculated features)
                 hidden_dim=64,          # Hidden dimension in the transformer
                 num_heads=4,            # Number of attention heads
                 num_layers=2,           # Number of transformer layers
                 dropout=0.1,            # Dropout rate
                 max_seq_length=10):     # Maximum number of tokens (elements + calculated features)
        super(ChemicalTransformer, self).__init__()
        
        # Input projection (from raw features to hidden dimension)
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # Standard practice: 4x hidden_dim
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression head - combines calculated features with element representations
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Combines calculated features + weighted elements
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Single output for regression
        )
        
        # Save parameters for later use
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
    def create_padding_mask(self, tokens):
        """Create mask for padding tokens"""
        # Sum across feature dimension - if all zeros, it's a padding token
        # Shape: [batch_size, seq_length]
        return (tokens.sum(dim=2) != 0).float()
    
    def forward(self, x, element_percentages=None):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_length, feature_dim]
               - First n-1 tokens are elements
               - Last token is calculated features
            element_percentages: Optional tensor of element percentages [batch_size, num_elements]
                                If None, will extract from x[:,:-1,1]
        """
        # Add batch dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, seq_length, feature_dim]
            
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # If element_percentages not provided, extract from input tensor
        if element_percentages is None:
            # Extract percentages from column 1 of each element token (excluding calculated features)
            element_percentages = x[:,:-1,1]  # [batch_size, num_elements]
        
        # Create padding mask if needed (1 for real tokens, 0 for padding)
        padding_mask = self.create_padding_mask(x)
        
        # Convert padding mask to key padding mask format (False where tokens should be attended to)
        key_padding_mask = (padding_mask == 0)
        
        # Project input to hidden dimension
        projected_x = self.input_projection(x)  # [batch_size, seq_length, hidden_dim]
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            projected_x, 
            src_key_padding_mask=key_padding_mask
        )  # [batch_size, seq_length, hidden_dim]
        
        # Get calculated features representation (last non-padding token)
        # We can use the sum of padding mask to find the index of the calculated features token
        calc_features_idx = padding_mask.sum(dim=1).long() - 1  # [batch_size]
        
        calc_features_repr = []
        for i in range(batch_size):
            calc_features_repr.append(transformer_output[i, calc_features_idx[i]])
        calc_features_repr = torch.stack(calc_features_repr)  # [batch_size, hidden_dim]
        
        # Create weighted element representation based on percentages
        # Normalize percentages to sum to 1
        norm_percentages = element_percentages / (element_percentages.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted sum of element representations
        element_repr = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        for i in range(batch_size):
            # Only consider non-padding elements
            num_elements = int(padding_mask[i,:-1].sum().item())  # Exclude calculated features
            
            # Weighted sum of element representations
            for j in range(num_elements):
                element_repr[i] += norm_percentages[i,j] * transformer_output[i,j]
        
        # Combine calculated features and weighted element representation
        combined_repr = torch.cat([calc_features_repr, element_repr], dim=1)  # [batch_size, hidden_dim*2]
        
        # Pass through regression head
        output = self.regression_head(combined_repr)  # [batch_size, 1]
        
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# ChemicalTransformer class stays the same

# Example usage with proper batch training
if __name__ == "__main__":
    from tokenizer import LM_Tokenizer
    from dataloader import train_loader  # Assuming this is your dataset or dataloader

    torch.manual_seed(42)  # For reproducibility
    
    # Initialize tokenizer
    tokenizer = LM_Tokenizer()
    
    # Set hyperparameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    
    # Get sample data to determine input dimensions
    sample_composition = "Sn14.0Bi36In25.0Ag25.0"
    sample_target = 0.1
    sample_tokens, _ = tokenizer.encode(composition=sample_composition, target=sample_target)
    
    # Initialize model
    model = ChemicalTransformer(
        feature_dim=sample_tokens.shape[1],
        hidden_dim=64,
        num_heads=4,
        num_layers=2, 
        dropout=0.1,
        max_seq_length=sample_tokens.shape[0]
    )
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs:")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (compositions, targets) in enumerate(train_loader):
            # Process each composition in the batch
            batch_tokens = []
            batch_targets = []
            
            for comp, target in zip(compositions, targets):
                tokens, target_tensor = tokenizer.encode(composition=comp, target=target)
                batch_tokens.append(tokens)
                batch_targets.append(target_tensor)
            
            # Stack into batch tensors
            # Make sure all sequences have the same length (pad if necessary)
            max_seq_len = max(tokens.shape[0] for tokens in batch_tokens)
            feature_dim = batch_tokens[0].shape[1]
            
            # Create padded batch
            padded_batch = torch.zeros(len(batch_tokens), max_seq_len, feature_dim)
            for i, tokens in enumerate(batch_tokens):
                padded_batch[i, :tokens.shape[0], :] = tokens
            
            # Stack targets
            batch_targets = torch.stack(batch_targets)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(padded_batch)
            
            # Calculate loss
            loss = criterion(predictions.squeeze(), batch_targets.squeeze())
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print progress for every 5 batches
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # Print epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} completed, Avg Loss: {avg_loss:.6f}")
        
        # Evaluate on a sample after each epoch
        model.eval()
        with torch.no_grad():
            sample_tokens, sample_target = tokenizer.encode(composition=sample_composition, target=sample_target)
            prediction = model(sample_tokens)
            print(f"Sample evaluation - Prediction: {prediction.item():.4f}, Target: {sample_target.item():.4f}")
    
    print("\nTraining completed!")
    
    # Final evaluation on a few test samples
    model.eval()
    test_compositions = [
        "Sn14.0Bi36In25.0Ag25.0",
        "Ga10.0In20.0Sn70.0",
        "Bi40.0Sn20.0In40.0"
    ]
    test_targets = [0.1, 0.2, 0.3]
    
    print("\nFinal model evaluation:")
    with torch.no_grad():
        for comp, target in zip(test_compositions, test_targets):
            tokens, target_tensor = tokenizer.encode(composition=comp, target=target)
            prediction = model(tokens)
            print(f"Composition: {comp}, Prediction: {prediction.item():.4f}, Target: {target}")