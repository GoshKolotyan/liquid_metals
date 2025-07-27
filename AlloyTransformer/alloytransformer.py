import torch
from torch.nn.init import xavier_uniform_, zeros_
from torch.nn import (
    Module,
    Linear,
    ModuleList,
    Dropout,
    LayerNorm,
    MultiheadAttention,
    Sequential,
    ReLU,
    Parameter,
)


class PropertyFocusedAttention(Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, property_bias: bool):
        super(PropertyFocusedAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # Projection matrices
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

        # 🔧 FIXED: Property bias initialization (was overwriting itself)
        if property_bias:
            # Enhanced property bias for better attention patterns
            self.register_buffer('melting_point_threshold', torch.tensor(1000.0))
            self.property_bias = Parameter(torch.zeros(num_heads, 6, 6))
            # Initialize with slight bias toward elements with high melting points
            torch.nn.init.normal_(self.property_bias, mean=0.0, std=0.01)  # Reduced std for stability
        else:
            self.property_bias = None

        self.dropout = Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input [batch_size, seq_len, d_model]
            attention_mask: Boolean [batch_size, seq_len] - True = real, False = padding
        """
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply property bias if specified
        if self.property_bias is not None:
            # Handle variable sequence lengths for property bias
            if seq_len <= self.property_bias.shape[-1]:
                bias = self.property_bias[:, :seq_len, :seq_len].unsqueeze(0)
                scores = scores + bias
            # If sequence is longer, skip bias (shouldn't happen with current data)

        # 🚨 CRITICAL FIX: Apply attention mask
        if attention_mask is not None:
            # Expand mask for multi-head attention
            # attention_mask: [batch, seq_len] → [batch, 1, 1, seq_len]
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(1)
            
            # Apply mask: set padded positions to -inf before softmax
            scores = scores.masked_fill(~mask_expanded, float('-inf'))

        # Apply softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final projection
        output = self.out_proj(output)
        return output


class AlloyTransformer(Module):
    def __init__(self, feature_dim: int, d_model: int, num_head: int, num_transformer_layers: int, 
                 num_regression_head_layers: int, dropout: float, num_positions: int, dim_feedforward: int, 
                 use_property_focus: bool, debug_mode: bool = False):
        super(AlloyTransformer, self).__init__()
        self.d_model = d_model
        self.num_positions = num_positions
        self.debug_mode = debug_mode  # 🔧 ADDED: Debug mode control

        # input dim is (batch_size, 6, 5)->(batch_size, 6, d_model)
        self.feature_embeddings = Linear(feature_dim, d_model)

        # learnable role embeddings (batch_size, 6, d_model)
        self.role_embeddings = Parameter(torch.zeros(1, num_positions, d_model))
        xavier_uniform_(self.role_embeddings)

        self.layers = ModuleList()
        for _ in range(num_transformer_layers):
            if use_property_focus:
                attention = PropertyFocusedAttention(d_model=d_model, num_heads=num_head, dropout=dropout, property_bias=True)
            else:
                attention = MultiheadAttention(embed_dim=d_model, num_heads=num_head, dropout=dropout, batch_first=True)

            # full encoder layer
            self.layers.append(
                ModuleList(
                    [
                        LayerNorm(d_model),
                        attention,
                        Dropout(dropout),
                        LayerNorm(d_model),
                        Sequential(
                            Linear(d_model, dim_feedforward),
                            ReLU(),
                            Dropout(dropout),
                            Linear(dim_feedforward, d_model),
                        ),
                        Dropout(dropout),
                    ]
                )
            )

        # final layer norm
        self.final_norm = LayerNorm(d_model)

        regression_layers = []
        for _ in range(num_regression_head_layers-1):
                regression_layers.extend([
                Linear(d_model, d_model),
                ReLU(),
                Dropout(dropout)
            ])

        regression_layers.append(Linear(d_model, 1))

        self.regression_head = Sequential(*regression_layers)

        # init weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, feature_dim]
            attention_mask: Boolean tensor [batch_size, seq_len] - True for real data, False for padding
        """
        
        # 🚨 CRITICAL FIX: Handle shuffling with attention mask
        if self.training and attention_mask is not None:
            shuffled_x = x.clone()
            shuffled_mask = attention_mask.clone()
            
            if self.debug_mode:
                print(50 * "==")
                print(f"Input x before shuffling shape: {x.shape}")
                print(f"Input attention_mask before shuffling: {attention_mask}")
            
            batch_size = x.shape[0]
            
            # 🔧 IMPROVED: More robust shuffling with safety checks
            # Shuffle each sample individually to preserve mask consistency
            for b in range(batch_size):
                # Find how many real element positions this sample has
                real_positions = attention_mask[b].sum().item()
                if real_positions > 1:  # If more than just features
                    num_elements = real_positions - 1  # Subtract 1 for features row
                    if num_elements > 1 and num_elements <= 5:  # Safety bounds
                        # Generate shuffled indices for elements only
                        element_indices = torch.randperm(num_elements)
                        
                        # Apply shuffling to elements (preserve features and padding)
                        shuffled_x[b, :num_elements, :] = x[b, element_indices, :]
                        shuffled_mask[b, :num_elements] = attention_mask[b, element_indices]
            
            if self.debug_mode:
                print(f"Shuffled x shape: {shuffled_x.shape}")
                print(f"Shuffled attention_mask: {shuffled_mask}")
                print(20*"==", "Shuffled is X" ,20*"==")
                print(shuffled_x)
                print(50 * "==")
            
            x = shuffled_x
            attention_mask = shuffled_mask
        else:
            if self.debug_mode:
                print(50 * "==")
                print(f"Input x shape: {x.shape}")
                if attention_mask is not None:
                    print(f"Input attention_mask: {attention_mask}")
                print(50 * "==")

        # Apply feature embeddings
        x = self.feature_embeddings(x)
        if self.debug_mode:
            print(f"x shape after feature embedding: {x.shape}")

        # 🚨 CRITICAL FIX: Apply role embeddings (was commented out!)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Truncate or expand role embeddings to match actual sequence length
        if seq_len <= self.num_positions:
            role_embeddings = self.role_embeddings[:, :seq_len, :].expand(batch_size, -1, -1)
        else:
            # If sequence is longer than expected, pad role embeddings
            extra_positions = seq_len - self.num_positions
            extra_embeddings = torch.zeros(1, extra_positions, self.d_model, 
                                         device=self.role_embeddings.device,
                                         dtype=self.role_embeddings.dtype)
            extended_role_embeddings = torch.cat([self.role_embeddings, extra_embeddings], dim=1)
            role_embeddings = extended_role_embeddings.expand(batch_size, -1, -1)
        
        if self.debug_mode:
            print(f"role_embeddings shape: {role_embeddings.shape}")
        
        x = x + role_embeddings
        if self.debug_mode:
            print(f"x shape after adding role embeddings: {x.shape}")

        # Process through transformer layers WITH attention masking
        for i, (layer_norm1, attn, dropout1, layer_norm2, ffn, dropout2) in enumerate(self.layers):
            if self.debug_mode:
                print(50 * "==")
            norm_x = layer_norm1(x)
            
            # 🚨 CRITICAL FIX: Apply attention with masking
            if isinstance(attn, PropertyFocusedAttention):
                if self.debug_mode:
                    print(18*"==", f"Using PropertyFocusedAttention", 16*"==")
                attn_output = attn(norm_x, attention_mask=attention_mask)
                if self.debug_mode:
                    print(f"Layer {i+1} attention output shape: {attn_output.shape}")
            else:
                if self.debug_mode:
                    print(18*"==", f"Using MultiheadAttention", 19*"==")
                if attention_mask is not None:
                    # Convert boolean mask to key_padding_mask format
                    # PyTorch expects True = positions to IGNORE, so invert
                    key_padding_mask = ~attention_mask
                    attn_output, _ = attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask)
                else:
                    attn_output, _ = attn(norm_x, norm_x, norm_x)
                if self.debug_mode:
                    print(f"Layer {i+1} attention output shape: {attn_output.shape}")
            
            # Residual connection
            x = x + dropout1(attn_output)
            if self.debug_mode:
                print(f"Layer {i+1} after residual connection shape: {x.shape}")
            
            # FFN with pre-norm
            norm_x = layer_norm2(x)
            if self.debug_mode:
                print(f"Layer {i+1} ffn pre-norm shape: {norm_x.shape}")
            ffn_output = ffn(norm_x)
            if self.debug_mode:
                print(f"Layer {i+1} ffn output shape: {ffn_output.shape}")
            
            # Residual connection
            x = x + dropout2(ffn_output)
            if self.debug_mode:
                print(f"Layer {i+1} output shape: {x.shape}")

        # Final layer norm
        x = self.final_norm(x)
        if self.debug_mode:
            print(f"Final output shape: {x.shape}")

        # 🚨 CRITICAL FIX: Masked global pooling instead of simple mean
        if attention_mask is not None:
            # Only average over real positions (ignore padding)
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]
            
            # Zero out padded positions
            masked_x = x * mask_expanded  # [batch, seq, d_model]
            
            # Sum over sequence dimension
            sum_x = torch.sum(masked_x, dim=1)  # [batch, d_model]
            
            # Count number of real positions per sample
            mask_sum = torch.sum(mask_expanded, dim=1)  # [batch, 1]
            
            # Average only over real positions (avoid division by zero)
            mask_sum = torch.clamp(mask_sum, min=1.0)
            pooled = sum_x / mask_sum.squeeze(-1)  # [batch, d_model]
            
            if self.debug_mode:
                print(f"Masked pooled output shape: {pooled.shape}")
                print(f"Average pooling over real positions only")
        else:
            # Fallback: regular mean pooling
            pooled = torch.mean(x, dim=1)
            if self.debug_mode:
                print(f"Regular pooled output shape: {pooled.shape}")

        # Final prediction
        melting_point = self.regression_head(pooled).squeeze(-1)
        return melting_point


if __name__ == "__main__":
    from dataloader import LM_Dataset, collate_fn
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset=LM_Dataset("./Data/example.csv"), collate_fn=collate_fn, batch_size=1
    )
    
    # 🚨 CRITICAL CHANGES: Reduced model size for extrapolation
    model = AlloyTransformer(
        feature_dim=5,
        d_model=128,           # 🔧 REDUCED from 1024 to 128
        num_head=4,            # 🔧 REDUCED from 16 to 4
        num_transformer_layers=2,  # 🔧 REDUCED from 3 to 2
        num_regression_head_layers=2,  # 🔧 REDUCED from 3 to 2
        dropout=0.4,           # 🔧 INCREASED from 0.1 to 0.4
        num_positions=6,
        dim_feedforward=256,   # 🔧 REDUCED from 512 to 256
        use_property_focus=True,
        debug_mode=False        # 🔧 ADDED: Enable debugging for testing
    )

    # Calculate and display model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"MODEL PARAMETER COUNT")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Expected parameter range for extrapolation: < 1,000,000")
    print(f"Status: {'✅ GOOD' if total_params < 1000000 else '❌ TOO LARGE'}")
    print(f"{'='*60}\n")

    for batch in dataloader:
        composition_tensor, target_tensor, attention_mask = batch
        print(f'Features shape: {composition_tensor.shape}')
        print(f'Target shape: {target_tensor.shape}')
        print(f'Attention mask shape: {attention_mask.shape}')
        print(f'Attention mask: {attention_mask}')

        output = model(composition_tensor, attention_mask)
        print(50 * "==")
        print(f"Model output: {output.item()}")
        # break  # Only test first batch

    print(f"\n🎯 Model is ready for pentanary extrapolation training!")
    print(f"Parameters: {total_params:,} (target: <1M)")
    print(f"Architecture: {model.d_model}d × {len(model.layers)}L × {model.layers[0][1].num_heads if hasattr(model.layers[0][1], 'num_heads') else 'N/A'}H")