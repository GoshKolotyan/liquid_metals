import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CustomChemicalAttention(nn.Module):
    """
    Custom attention mechanism for chemical data with detailed logging
    """
    def __init__(
        self,
        embed_dim=256,          # Same as hidden_dim in ChemicalTransformer
        num_heads=16,           # Number of attention heads
        head_dim=None,          # If None, calculated from embed_dim/num_heads
        dropout=0.1,            # Attention dropout rate
        temperature=1.0,        # Temperature for scaling attention distribution
        causal_mask=False,      # For chemical data, we typically don't need causal masking
        position_encoding=None, # "none", "absolute", "relative", "rope", "alibi"
        verbose=True,           # Control logging
        debug_level=1,          # Log detail level (1=normal, 2=detailed, 3=very detailed)
    ):
        super(CustomChemicalAttention, self).__init__()
        
        # Store hyperparameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        self.temperature = temperature
        self.use_causal_mask = causal_mask
        self.position_encoding = position_encoding
        self.verbose = verbose
        self.debug_level = debug_level
        
        # Calculate total dimension for all heads
        self.total_head_dim = self.head_dim * num_heads
        
        self.log(f"[INIT] Creating CustomChemicalAttention with:", 1)
        self.log(f"  - embed_dim={embed_dim} (Embedding dimension size)", 1)
        self.log(f"  - num_heads={num_heads} (Number of attention heads)", 1)
        self.log(f"  - head_dim={self.head_dim} (Dimension per head)", 1)
        self.log(f"  - total_head_dim={self.total_head_dim} (Total attention dimension)", 1)
        self.log(f"  - dropout={dropout} (Attention dropout rate)", 1)
        self.log(f"  - temperature={temperature} (Attention temperature scaling)", 1)
        
        # Define projection matrices for Q, K, V
        self.q_proj = nn.Linear(embed_dim, self.total_head_dim)
        self.k_proj = nn.Linear(embed_dim, self.total_head_dim)
        self.v_proj = nn.Linear(embed_dim, self.total_head_dim)
        self.log("Created Q, K, V projection layers", 1)
        
        # Output projection
        self.out_proj = nn.Linear(self.total_head_dim, embed_dim)
        self.log(f"Created output projection: {self.total_head_dim} → {embed_dim}", 1)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.log(f"Created dropout layers with rate {dropout}", 1)
        
        # Position encoding components (if used)
        if position_encoding == "absolute":
            self.pos_embed = nn.Parameter(torch.zeros(1, 512, embed_dim))
            nn.init.normal_(self.pos_embed, std=0.02)
            self.log("Created absolute positional embeddings", 1)
        elif position_encoding == "alibi":
            # ALiBi slopes per head
            slopes = torch.Tensor([2**(-8 * (i / num_heads)) for i in range(num_heads)])
            self.register_buffer("alibi_slopes", slopes.view(num_heads, 1, 1))
            self.log("Created ALiBi positional bias", 1)
        
        # Scaling factor for attention scores
        self.scaling = self.head_dim ** -0.5
        self.log(f"Attention scaling factor: {self.scaling}", 2)
        
        # Initialize causal mask if needed
        if self.use_causal_mask:
            mask = torch.tril(torch.ones(512, 512)).view(1, 1, 512, 512)
            self.register_buffer("causal_mask_buffer", mask)
            self.log("Created causal masking buffer", 1)
        
        # Initialize weights
        self._init_weights()
        self.log("Initialized weights with Xavier uniform and zeros for biases", 1)
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers"""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.normal_(p)
                if self.debug_level >= 3:
                    self.log(f"  - Initialized {name} with normal uniform, shape: {p.shape}", 3)
            elif 'bias' in name:
                nn.init.zeros_(p)
                if self.debug_level >= 3:
                    self.log(f"  - Initialized {name} with zeros, shape: {p.shape}", 3)
            

    
    def log(self, message, level=1):
        """Helper function to print logs only when verbose is True and at appropriate debug level"""
        if self.verbose and level <= self.debug_level:
            print(message)

    def visualize_attention(self, attn_weights):
        """Helper to visualize attention weights (if matplotlib is available)"""
        if not self.verbose:
            return
            
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Convert to numpy for visualization
            weights = attn_weights.detach().cpu().numpy()
            
            plt.figure(figsize=(10, 8))
            plt.imshow(weights, cmap='viridis')
            plt.colorbar()
            plt.title('Element Attention Weights')
            plt.xlabel('Element Position')
            plt.ylabel('Query Position')
            plt.savefig('attention_weights.png')
            self.log(f"Saved attention visualization to attention_weights.png", 1)
            plt.close()
        except ImportError:
            self.log("Matplotlib not available for attention visualization", 1)
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True):
        """
        Forward pass through custom chemical attention mechanism
        Compatible with nn.MultiheadAttention interface
        """
        self.log("\n[ATTENTION] Running CustomChemicalAttention forward pass", 1)
        
        # Get shapes
        batch_size, query_len, _ = query.shape
        _, key_len, _ = key.shape
        self.log(f"[ATTENTION] Input shapes:", 1)
        self.log(f"  - Query: {query.shape}", 1)
        self.log(f"  - Key: {key.shape}", 1)
        self.log(f"  - Value: {value.shape}", 1)
        
        if key_padding_mask is not None:
            self.log(f"  - Key padding mask: {key_padding_mask.shape}", 1)
            self.log(f"  - Masked positions count: {key_padding_mask.sum().item()}", 2)
        
        # STEP 1: Project inputs to queries, keys, and values
        self.log("[ATTENTION-STEP 1] Projecting queries, keys, and values", 1)
        q = self.q_proj(query)  # [batch_size, query_len, total_head_dim]
        k = self.k_proj(key)    # [batch_size, key_len, total_head_dim]
        v = self.v_proj(value)  # [batch_size, key_len, total_head_dim]
        
        self.log(f"  - Projected Q shape: {q.shape}", 2)
        self.log(f"  - Projected K shape: {k.shape}", 2)
        self.log(f"  - Projected V shape: {v.shape}", 2)
        
        # STEP 2: Reshape for multi-head attention
        self.log("[ATTENTION-STEP 2] Reshaping for multi-head attention", 1)
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        
        self.log(f"  - Reshaped Q: {q.shape}", 2)
        self.log(f"  - Reshaped K: {k.shape}", 2)
        self.log(f"  - Reshaped V: {v.shape}", 2)
        
        # STEP 3: Compute attention scores (scaled dot-product)
        self.log("[ATTENTION-STEP 3] Computing scaled dot-product attention scores", 1)
        self.log(f"  - Applying scaling factor: {self.scaling}", 2)
        
        # (q * scaling) @ k.transpose = [batch_size, num_heads, query_len, key_len]
        attn_weights = (q * self.scaling) @ k.transpose(-2, -1)
        self.log(f"  - Raw attention weights shape: {attn_weights.shape}", 2)
        
        if self.debug_level >= 3:
            self.log(f"  - Attention weights statistics: min={attn_weights.min().item():.4f}, max={attn_weights.max().item():.4f}, mean={attn_weights.mean().item():.4f}", 3)
        
        # STEP 4: Apply padding mask if provided
        if key_padding_mask is not None:
            self.log("[ATTENTION-STEP 4] Applying key padding mask", 1)
            # Convert to [batch_size, 1, 1, key_len]
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            
            self.log(f"  - Expanded mask shape: {expanded_mask.shape}", 2)
            self.log(f"  - Filling masked positions with -inf", 2)
            
            attn_weights = attn_weights.masked_fill(expanded_mask, float('-inf'))
            self.log(f"  - Masked attention weights shape: {attn_weights.shape}", 2)
        else:
            self.log("[ATTENTION-STEP 4] No key padding mask applied", 1)
        
        # STEP 5: Apply temperature scaling
        if self.temperature != 1.0:
            self.log(f"[ATTENTION-STEP 5] Applying temperature scaling: {self.temperature}", 1)
            attn_weights = attn_weights / self.temperature
            self.log(f"  - Scaled attention weights by temperature factor: {self.temperature}", 2)
        else:
            self.log("[ATTENTION-STEP 5] No temperature scaling applied (temperature=1.0)", 1)
        
        # STEP 6: Apply softmax to get attention probabilities
        self.log("[ATTENTION-STEP 6] Applying softmax to get attention probabilities", 1)
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        if self.debug_level >= 3:
            valid_probs = attn_probs[~torch.isnan(attn_probs)]
            if len(valid_probs) > 0:
                self.log(f"  - Attention probability statistics: min={valid_probs.min().item():.4f}, max={valid_probs.max().item():.4f}, mean={valid_probs.mean().item():.4f}", 3)
            else:
                self.log("  - WARNING: All attention probabilities are NaN!", 3)
        
        # STEP 7: Apply dropout to attention probabilities
        self.log("[ATTENTION-STEP 7] Applying dropout to attention probabilities", 1)
        attn_probs = self.attn_dropout(attn_probs)
        self.log(f"  - Applied dropout with rate: {self.attn_dropout.p}", 2)
        
        # STEP 8: Apply attention to values
        self.log("[ATTENTION-STEP 8] Applying attention weights to values", 1)
        # [batch_size, num_heads, query_len, key_len] @ [batch_size, num_heads, key_len, head_dim]
        context = torch.matmul(attn_probs, v)  # [batch_size, num_heads, query_len, head_dim]
        self.log(f"  - Context vectors shape: {context.shape}", 2)
        
        # STEP 9: Reshape back to original dimensions
        self.log("[ATTENTION-STEP 9] Reshaping back to original dimensions", 1)
        # Transpose and reshape: [batch_size, query_len, num_heads*head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.total_head_dim)
        self.log(f"  - Reshaped context shape: {context.shape}", 2)
        
        # STEP 10: Apply output projection
        self.log("[ATTENTION-STEP 10] Applying output projection", 1)
        attn_output = self.out_proj(context)  # [batch_size, query_len, embed_dim]
        attn_output = self.output_dropout(attn_output)
        self.log(f"  - Final output shape: {attn_output.shape}", 2)
        
        # STEP 11: Visualize attention if in debug mode
        if self.debug_level >= 2 and need_weights:
            self.log("[ATTENTION-STEP 11] Visualizing attention weights", 2)
            self.visualize_attention(attn_probs[0, 0])
        
        # Return output and optionally attention weights
        if need_weights:
            self.log("[ATTENTION] Returning output and attention weights", 1)
            return attn_output, attn_probs
        else:
            self.log("[ATTENTION] Returning only output (no attention weights)", 1)
            return attn_output, None


if __name__ == "__main__":
    from torch import tensor
    from pprint import pprint
    
    print("\n" + "="*80)
    print("CUSTOM CHEMICAL ATTENTION DEMO")
    print("="*80)
    
    # Sample input tensor
    input_tensor = tensor(
        [
            [
                [5.0000e+01, 1.3300e-01, 1.4500e+02, 1.9600e+00, 2.3193e+02],
                [8.3000e+01, 5.0000e-01, 1.6300e+02, 2.0200e+00, 2.7136e+02],
                [4.8000e+01, 1.0000e-01, 1.5800e+02, 1.6900e+00, 3.2107e+02],
                [8.2000e+01, 2.6700e-01, 1.5400e+02, 1.8700e+00, 3.2746e+02],
                [9.9579e+00, 0.0000e+00, 5.3623e-02, 3.9681e-02, 1.1477e-01]
         ]
        ]
    )
    print("Input tensor is")    
    pprint(input_tensor)
    
    # Extract components
    elements = input_tensor[:, :-1, :]  # All but last row
    calc_features = input_tensor[:, -1, :]  # Last row
    
    print(f"\nElements are")
    pprint(elements)
    
    element_ids = elements[:, :, 0:1]  # First column: element ID
    percentages = elements[:, :, 1:2]  # Second column: percentage
    element_features = elements[:, :, 2:5]  # Last 3 columns: other features
    
    print(f"\nElements Ids")
    pprint(element_ids)
    
    print(f"\nPercentages are")
    pprint(percentages)
    
    print(f"\nElements Features")
    pprint(element_features)
    
    print(f"\nCalculated Features are")
    pprint(calc_features)
    
    # Create the padding mask (1 for valid elements, 0 for padding)
    padding_mask = (elements.sum(dim=2) != 0).float()  # [batch, seq_len-1]
    print(f"\nPadding mask")
    pprint(padding_mask)
    
    # Create attention mask (True for positions to mask)
    attn_mask = (padding_mask == 0)  # [batch, seq_len-1]
    print(f"\nAttention mask (True = masked positions)")
    pprint(attn_mask)
    
    # Projection layers
    element_projection = nn.Linear(1, 64)  # For element ID
    percentage_projection = nn.Linear(1, 64)  # For percentage
    feature_projection = nn.Linear(3, 128)  # For the remaining 3 features
    calc_features_projection = nn.Linear(5, 256)  # For calculated features
    
    # Project features
    print("\n" + "="*80)
    print("PROJECTING FEATURES")
    print("="*80)
    
    id_embeds = element_projection(element_ids)  # [batch, seq_len-1, 64]
    percentage_embeds = percentage_projection(percentages)  # [batch, seq_len-1, 64]
    feature_embeds = feature_projection(element_features)  # [batch, seq_len-1, 128]
    
    print(f"ID embeddings shape: {id_embeds.shape}")
    print(f"Percentage embeddings shape: {percentage_embeds.shape}")
    print(f"Feature embeddings shape: {feature_embeds.shape}")
    
    # Combine element embeddings
    element_embeds = torch.cat([id_embeds, percentage_embeds, feature_embeds], dim=2)  # [batch, seq_len-1, 256]
    print(f"Combined element embeddings shape: {element_embeds.shape}")
    
    # Project calculated features
    calc_features_embed = calc_features_projection(calc_features).unsqueeze(1)  # [batch, 1, 256]
    print(f"Calculated features embedding shape: {calc_features_embed.shape}")
    
    # Initialize custom attention
    print("\n" + "="*80)
    print("INITIALIZING CUSTOM CHEMICAL ATTENTION")
    print("="*80)
    
    chem_attention = CustomChemicalAttention(
        embed_dim=256,
        num_heads=4,
        head_dim=16,
        dropout=0.1,
        temperature=0.8,
        verbose=True,
        debug_level=3
    )
    
    # Prepare queries, keys, and values for attention
    print("\n" + "="*80)
    print("APPLYING ATTENTION MECHANISM")
    print("="*80)
    
    # Use calculated features as query to attend to elements
    query = calc_features_embed  # [batch, 1, 256]
    key = element_embeds         # [batch, seq_len-1, 256]
    value = element_embeds       # [batch, seq_len-1, 256]
    
    # Apply attention
    attended_elements, attention_weights = chem_attention(
        query=query,
        key=key,
        value=value,
        key_padding_mask=attn_mask,
        need_weights=True
    )
    
    print("\n" + "="*80)
    print("ATTENTION RESULTS")
    print("="*80)
    
    print(f"Attended elements shape: {attended_elements.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Combine attended elements with calculated features
    combined_repr = torch.cat([calc_features_embed, attended_elements], dim=2)  # [batch, 1, 512]
    print(f"Combined representation shape: {combined_repr.shape}")
    
    # Example fusion layer
    fusion_layer = nn.Linear(512, 256)
    final_repr = fusion_layer(combined_repr)  # [batch, 1, 256]
    print(f"Final fused representation shape: {final_repr.shape}")
    
    # Example regression head
    regression_head = nn.Linear(256, 1)
    prediction = regression_head(final_repr)  # [batch, 1, 1]
    print(f"Final prediction shape: {prediction.shape}")
    print(f"Prediction value: {prediction.item():.4f}")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)