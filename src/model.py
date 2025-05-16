import torch
import torch.nn as nn
import torch.nn.functional as F

class ChemicalTransformer(nn.Module):
    def __init__(self, 
                 feature_dim=5,       
                 hidden_dim=128,      
                 num_heads=64,         
                 num_layers=2,        
                 dropout=0.1,         
                 max_seq_length=512,  
                 num_elements=118,
                 verbose=False,
                 debug_level=1):   # Added verbose parameter to control logging
        super(ChemicalTransformer, self).__init__()
        
        self.verbose = verbose  # Store verbose flag
        self.debug_level = debug_level  # Different levels of verbosity (1=normal, 2=detailed, 3=very detailed)
        
        print(f"[INIT] Creating ChemicalTransformer with:")
        print(f"  - feature_dim={feature_dim} (Input feature dimensions)")
        print(f"  - hidden_dim={hidden_dim} (Internal representation size)")
        print(f"  - num_heads={num_heads} (Attention heads in transformer)")
        print(f"  - num_layers={num_layers} (Transformer encoder layers)")
        print(f"  - dropout={dropout} (Dropout rate for regularization)")
        
        # Original model expects 
        # - First column (index 0): element_id (atomic number or element identifier)
        # - Second column (index 1): percentage (concentration in compound)
        # - Remaining 3 columns (indices 2,3,4): other features (properties of elements)
        
        # Feature projection layers - convert raw features into learned embeddings
        self.element_projection = nn.Linear(1, 128)  # For element ID
        self.percentage_projection = nn.Linear(1, 128)  # For percentage
        self.feature_projection = nn.Linear(3, 256)  # For the remaining 3 features
        self.log("Created feature projection layers: element_id(1→128), percentage(1→128), features(3→256)")
        
        # Separate projection for calculated features token (global properties token)
        self.calc_features_projection = nn.Linear(feature_dim, hidden_dim)
        self.log("Created calculated features projection layer")
        
        # Transformer encoder layer (main reasoning component)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False)
        self.log(f"Created transformer encoder with {num_layers} layers and {num_heads} attention heads")
        
        # Element attention mechanism (helps focus on relevant elements)
        self.element_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads//2,
            dropout=dropout,
            batch_first=True,
        )
        self.log("Created element attention mechanism with 8 heads")
        
        # Feature fusion layer (combines different representations)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.log("Created feature fusion layer")
        
        # Regression head (final prediction component)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.log("Created regression head for final prediction")
        
        # Save parameters for later use
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # Initialize weights
        self._init_weights()
        print("[INIT] Model initialization complete")
        
    def _init_weights(self):
        print("[INIT] Initializing model weights using Xavier uniform for weights and zeros for biases")
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
                if self.debug_level >= 2:
                    self.log(f"  - Initialized {name} with Xavier uniform, shape: {p.shape}")
            elif 'bias' in name:
                nn.init.zeros_(p)
                if self.debug_level >= 2:
                    self.log(f"  - Initialized {name} with zeros, shape: {p.shape}")
    
    def log(self, message, level=1):
        """Helper function to print logs only when verbose is True and at appropriate debug level"""
        if self.verbose and level <= self.debug_level:
            print(message)
    
    def visualize_attention(self, attn_weights, padding_mask=None):
        """Helper to visualize attention weights (if matplotlib is available)"""
        if not self.verbose:
            return
            
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Convert to numpy for visualization
            weights = attn_weights.detach().cpu().numpy()
            print(weights)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(weights, cmap='viridis')
            plt.colorbar()
            plt.title('Element Attention Weights')
            plt.xticks(np.arange(4), ['1 pos', "2 pos", '3 pos', '4 pos'])
            plt.xlabel('Element Position')
            plt.ylabel('Attention Head')
            plt.show()
        except ImportError:
            self.log("Matplotlib not available for attention visualization")
    
    def forward(self, x):
        """
        Forward pass through the ChemicalTransformer model
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, feature_dim]
               - The last row (seq_length-1) contains calculated/global features
               - Each row before that represents an element in the chemical compound
               - Each element has features: [element_id, percentage, feature1, feature2, feature3]
        
        Returns:
            Predicted property value of shape [batch_size, 1]
        """
        self.log(f"[FORWARD] Input tensor shape: {x.shape}", 1)
        self.log(f"[FORWARD] Input tensor data type: {x.dtype}", 2)
        if self.debug_level >= 3:
            self.log(f"[FORWARD] Input tensor values (first batch):\n{x[0]}", 3)
            
        batch_size = x.size(0)
        seq_length = x.size(1)
        self.log(f"[FORWARD] Batch size: {batch_size}, Sequence length: {seq_length}", 1)
        
        # STEP 1: Split input into elements and calculated features
        self.log("\n[STEP 1] Splitting input into elements and calculated features", 1)
        self.log("  - Elements: all rows except the last one (chemical elements in compound)", 2)
        self.log("  - Calculated features: last row (global compound properties)", 2)
        
        elements = x[:, :-1, :]  # All but last row
        calc_features = x[:, -1, :]  # Last row
        self.log(f"[STEP 1] Elements shape : {elements.shape}, Calculated features shape: {calc_features.shape}", 1)
        
        if self.debug_level >= 3:
            self.log(f"[STEP 1] Elements sample (first batch, first element):\n{elements[0, 0]}", 3)
            self.log(f"[STEP 1] Calculated features sample (first batch):\n{calc_features[0]}", 3)
        
        # STEP 2: Extract element features
        self.log("\n[STEP 2] Extracting individual features from elements tensor", 1)
        self.log("  - Element IDs: First column (atomic numbers or identifiers)", 2)
        self.log("  - Percentages: Second column (concentration in compound)", 2)
        self.log("  - Element features: Last 3 columns (properties of elements)", 2)
        
        element_ids = elements[:, :, 0:1]          # First column: element ID
        percentages = elements[:, :, 1:2]          # Second column: percentage
        element_features = elements[:, :, 2:5]     # Last 3 columns: other features
        
        self.log(f"[STEP 2] Element IDs shape: {element_ids.shape}", 1)
        self.log(f"[STEP 2] Percentages shape: {percentages.shape}", 1)
        self.log(f"[STEP 2] Element features shape: {element_features.shape}", 1)
        
        if self.debug_level >= 3:
            self.log(f"[STEP 2] Element IDs sample (first batch):\n{element_ids}", 3)
            self.log(f"[STEP 2] Percentages sample (first batch):\n{percentages}", 3)
            self.log(f"[STEP 2] Element features sample (first batch):\n{element_features}", 3)
        
        # STEP 3: Create padding mask (1 for valid elements, 0 for padding)
        self.log("\n[STEP 3] Creating padding mask (1 for valid elements, 0 for padding)", 1)
        self.log("  - Purpose: To handle variable numbers of elements in compounds", 2)
        self.log("  - Method: Sum along feature dimension, if all zeros = padding token", 2)
        
        padding_mask = (elements.sum(dim=2) != 0).float()  # [batch, seq_len-1]
        self.log(f"[STEP 3] Padding mask shape: {padding_mask.shape}", 1)
        self.log(f"[STEP 3] Number of valid elements: {padding_mask.sum().item()}/{padding_mask.numel()}", 1)
        
        if self.debug_level >= 3:
            self.log(f"[STEP 3] Padding mask sample (first batch):\n{padding_mask[0]}", 3)
        
        # STEP 4: Project each feature separately
        self.log("\n[STEP 4] Projecting each feature type into embedding space", 1)
        self.log("  - Purpose: Convert raw features into learned embeddings", 2)
        self.log("  - Process: Apply linear transformations to each feature type", 2)
        
        id_embeds = self.element_projection(element_ids)            # [batch, seq_len-1, 32]
        percentage_embeds = self.percentage_projection(percentages) # [batch, seq_len-1, 32]
        
        self.log(f"[STEP 4] Element ID embeddings shape: {id_embeds.shape}", 1)
        self.log(f"[STEP 4] Percentage embeddings shape: {percentage_embeds.shape}", 1)
        
        self.log(f"[STEP 4] Element features before projection: {element_features.shape}", 1)
        feature_embeds = self.feature_projection(element_features)  # [batch, seq_len-1, 64]
        self.log(f"[STEP 4] Element features after projection: {feature_embeds.shape}", 1)
        
        if self.debug_level >= 3:
            self.log(f"[STEP 4] ID embeddings sample (first element, first 5 values):\n{id_embeds[0, 0, :5]}", 3)
            self.log(f"[STEP 4] Percentage embeddings sample (first element, first 5 values):\n{percentage_embeds[0, 0, :5]}", 3)
            self.log(f"[STEP 4] Feature embeddings sample (first element, first 5 values):\n{feature_embeds[0, 0, :5]}", 3)
        
        # STEP 5: Combine all element embeddings
        self.log("\n[STEP 5] Concatenating all element embeddings", 1)
        self.log("  - Purpose: Create unified representation for each element", 2)
        self.log("  - Process: Concatenate ID, percentage and feature embeddings", 2)
        
        element_embeds = torch.cat([id_embeds, percentage_embeds, feature_embeds], dim=2)  # [batch, seq_len-1, 128]
        self.log(f"[STEP 5] Combined element embeddings shape: {element_embeds.shape}", 1)
        
        # STEP 6: Project calculated features
        self.log("\n[STEP 6] Projecting calculated features", 1)
        self.log("  - Purpose: Create embedding for global compound properties", 2)
        self.log("  - Process: Project calculated features to hidden dimension space", 2)
        
        calc_features_embed = self.calc_features_projection(calc_features).unsqueeze(1)  # [batch, 1, 128]
        self.log(f"[STEP 6] Calculated features input shape: {calc_features.shape}", 1)
        self.log(f"[STEP 6] Calculated features embedding shape: {calc_features_embed.shape}", 1)
        
        # STEP 7: Combine all tokens for transformer input
        self.log("\n[STEP 7] Combining element embeddings and calculated features for transformer input", 1)
        self.log("  - Purpose: Create sequence for transformer to process", 2)
        self.log("  - Process: Concatenate element embeddings with calculated features", 2)
        
        transformer_input = torch.cat([element_embeds, calc_features_embed], dim=1)  # [batch, seq_len, 128]
        self.log(f"[STEP 7] Transformer input shape: {transformer_input.shape}", 1)
        
        # STEP 8: Add calculated features token to the padding mask
        self.log("\n[STEP 8] Updating padding mask to include calculated features token", 1)
        self.log("  - Purpose: Ensure calculated features token is always attended to", 2)
        self.log("  - Process: Add ones for calculated features position in padding mask", 2)
        
        full_padding_mask = torch.cat([padding_mask, torch.ones(batch_size, 1, device=x.device)], dim=1)
        self.log(f"[STEP 8] Full padding mask shape: {full_padding_mask.shape}", 1)
        
        # Convert to transformer mask format (True for padding positions)
        attn_mask = (full_padding_mask == 0)  # [batch, seq_len]
        self.log(f"[STEP 8] Attention mask shape: {attn_mask.shape}", 1)
        self.log(f"[STEP 8] Number of masked positions: {attn_mask.sum().item()}", 1)
        
        if self.debug_level >= 3:
            self.log(f"[STEP 8] Attention mask sample (first batch):\n{attn_mask[0]}", 3)
        
        # STEP 9: Pass through transformer encoder
        self.log("\n[STEP 9] Passing data through transformer encoder", 1)
        self.log("  - Purpose: Allow elements to interact and share information", 2)
        self.log("  - Process: Apply self-attention mechanism across elements", 2)
        
        transformer_output = self.transformer_encoder(
            transformer_input, 
            src_key_padding_mask=attn_mask
        )  # [batch, seq_len, hidden_dim]
        self.log(f"[STEP 9] Transformer output shape: {transformer_output.shape}", 1)
        
        if self.debug_level >= 3:
            self.log(f"[STEP 9] Transformer output sample (first batch, first token, first 5 values):\n{transformer_output[0, 0, :5]}", 3)
        
        # STEP 10: Get calculated features representation (last token)
        self.log("\n[STEP 10] Extracting calculated features representation from transformer output", 1)
        self.log("  - Purpose: Get enriched global compound representation", 2)
        self.log("  - Process: Extract the last token from transformer output", 2)
        
        calc_features_repr = transformer_output[:, -1, :]  # [batch, hidden_dim]
        self.log(f"[STEP 10] Calculated features representation shape: {calc_features_repr.shape}", 1)
        
        # STEP 11: Apply element attention to focus on important elements
        self.log("\n[STEP 11] Applying element attention to focus on important elements", 1)
        self.log("  - Purpose: Focus on elements that matter most for property prediction", 2)
        self.log("  - Process: Use calculated features as query to attend to elements", 2)
        
        element_repr = transformer_output[:, :-1, :]  # [batch, seq_len-1, hidden_dim]
        query = calc_features_repr.unsqueeze(1)  # [batch, 1, hidden_dim]
        self.log(f"[STEP 11] Element representations shape: {element_repr.shape}", 1)
        self.log(f"[STEP 11] Query shape: {query.shape}", 1)
        
        # Create element attention mask
        element_attn_mask = attn_mask[:, :-1]  # Exclude calculated features token
        self.log(f"[STEP 11] Element attention mask shape: {element_attn_mask.shape}", 1)
        
        # Apply attention
        self.log("[STEP 11] Applying multihead attention", 1)
        self.log("  - Query: Calculated features (what we're looking for)", 2)
        self.log("  - Keys/Values: Element representations (where to find information)", 2)
        
        attended_elements, attention_weights = self.element_attention(
            query,
            element_repr,
            element_repr,
            key_padding_mask=element_attn_mask
        )  # [batch, 1, hidden_dim]
        self.log(f"[STEP 11] Attended elements shape: {attended_elements.shape}", 1)
        self.log(f"[STEP 11] Attention weights shape: {attention_weights.shape}", 1)
        
        if self.debug_level >= 2:
            self.log("[STEP 11] Visualizing attention weights", 2)
            self.visualize_attention(attention_weights[0])  # Visualize first batch's attention
        
        attended_elements = attended_elements.squeeze(1)  # [batch, hidden_dim]
        self.log(f"[STEP 11] Attended elements after squeeze: {attended_elements.shape}", 1)
        
        # STEP 12: Combine calculated features and attended element representation
        self.log("\n[STEP 12] Combining calculated features and attended element representation", 1)
        self.log("  - Purpose: Create final representation using both global and element-specific info", 2)
        self.log("  - Process: Concatenate and fuse the two representations", 2)
        
        combined_repr = torch.cat([calc_features_repr, attended_elements], dim=1)  # [batch, hidden_dim*2]
        self.log(f"[STEP 12] Combined representation shape: {combined_repr.shape}", 1)
        
        fused_repr = self.fusion_layer(combined_repr)  # [batch, hidden_dim]
        self.log(f"[STEP 12] Fused representation shape: {fused_repr.shape}", 1)
        
        # STEP 13: Final prediction
        self.log("\n[STEP 13] Generating final prediction through regression head", 1)
        self.log("  - Purpose: Predict target property from fused representation", 2)
        self.log("  - Process: Apply final neural network layers to produce scalar output", 2)
        
        output = self.regression_head(fused_repr)  # [batch, 1]
        self.log(f"[STEP 13] Final output shape: {output.shape}", 1)
        
        if self.debug_level >= 2:
            self.log(f"[STEP 13] Prediction sample (first 3 batches): {output[:3].flatten().tolist()}", 2)
        
        return output

# Batch processing with detailed logging
def process_batch_with_logging(model, batch, verbose=True):
    """Process a batch with detailed logging"""
    if verbose:
        print("\n" + "="*80)
        print("BATCH PROCESSING")
        print("="*80)
    
    composition_tensor, target_tensor = batch
    
    if verbose:
        print(f"Batch composition tensor shape: {composition_tensor.shape}")
        print(f"Batch target tensor shape: {target_tensor.shape}")
        print(f"Number of compounds in batch: {composition_tensor.shape[0]}")
        print(f"Maximum number of elements per compound: {composition_tensor.shape[1]-1}")
        print(f"Number of features per element: {composition_tensor.shape[2]}")
        
        # Find actual number of elements in each compound (non-padding)
        elements_per_compound = (composition_tensor.sum(dim=2) != 0).sum(dim=1) - 1  # -1 for calc features
        print(f"Elements per compound: {elements_per_compound.tolist()}")
        
        # Show structure of first compound
        print("\nStructure of first compound in batch:")
        for i in range(composition_tensor.shape[1] - 1):  # Skip last row (calc features)
            element_row = composition_tensor[0, i]
            if element_row.sum() == 0:
                print(f"  Element {i}: [PADDING]")
            else:
                element_id = element_row[0].item()
                percentage = element_row[1].item()
                features = element_row[2:5].tolist()
                print(f"  Element {i}: ID={element_id}, Percentage={percentage:.4f}, Features={features}")
        
        # Show calculated features
        calc_features = composition_tensor[0, -1].tolist()
        print(f"  Calculated Features: {calc_features}")
        
        # Show targets
        print(f"\nTarget values: {target_tensor.flatten().tolist()}")
    
    # Forward pass
    if verbose:
        print("\nRunning forward pass through model...")
    
    # Save original verbosity setting
    original_verbose = model.verbose
    original_debug = model.debug_level
    
    # Use provided verbosity for this run
    model.verbose = verbose
    
    # Run the model
    output = model(composition_tensor)
    
    # Restore original verbosity
    model.verbose = original_verbose
    model.debug_level = original_debug
    
    if verbose:
        print("\nModel output vs targets:")
        for i in range(min(5, output.shape[0])):  # Show at most 5 examples
            predicted = output[i].item()
            actual = target_tensor[i].item()
            error = abs(predicted - actual)
            print(f"  Compound {i}: Predicted={predicted:.4f}, Actual={actual:.4f}, Error={error:.4f}")
    
    return output

# Instantiate model with your configuration
if __name__ == "__main__":
    # Import necessary modules
    from torchinfo import summary
    from dataloader import train_loader
    from torch.nn import L1Loss
    from torch.optim import Adam
    print("\n" + "="*80)
    print("INITIALIZING CHEMICAL TRANSFORMER MODEL")
    print("="*80)


    model = ChemicalTransformer(
        feature_dim=5,
        hidden_dim=256,
        num_heads=32,
        num_layers=2, 
        dropout=0.0,
        max_seq_length=512,
        verbose=True,
        debug_level=3,  # Set to 2 or 3 for more detailed logging,
    )

    loss = L1Loss()
    optimizer = Adam(model.parameters())
            
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    try:
        # Get model summary if torchinfo is available
        summary(model)
    except Exception as e:
        print(f"Could not generate model summary: {str(e)}")
    
    print("\n" + "="*80)
    print("PROCESSING BATCH")
    print("="*80)
    
    # Process a single batch from the training loader
    # Process a single batch from the training loader
    optimizer.zero_grad()  # Clear previous gradients
    for batch in train_loader:
        output = process_batch_with_logging(model, batch, verbose=True)
        # loss_ = loss(output, batch[1])
        # loss_.backward()    # Compute gradients
        # optimizer.step()    # Update model parameters
        break
        
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
        
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
