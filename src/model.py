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
                 num_elements=118):   
        super(ChemicalTransformer, self).__init__()
        
        # Original model expects 
        # - First column (index 0): element_id
        # - Second column (index 1): percentage
        # - Remaining 3 columns (indices 2,3,4): other features
        
        # Adjust the feature projection to match your data
        self.element_projection = nn.Linear(1, 32)  # For element ID
        self.percentage_projection = nn.Linear(1, 32)  # For percentage
        self.feature_projection = nn.Linear(3, 64)  # For the remaining 3 features
        
        # Separate projection for calculated features token
        self.calc_features_projection = nn.Linear(feature_dim, hidden_dim)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Element attention mechanism
        self.element_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Save parameters for later use
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Split input into elements and calculated features
        elements = x[:, :-1, :]  # All but last row
        calc_features = x[:, -1, :]  # Last row
        
        # Extract element features
        element_ids = elements[:, :, 0:1]          # First column: element ID
        percentages = elements[:, :, 1:2]          # Second column: percentage
        element_features = elements[:, :, 2:5]     # Last 3 columns: other features
        
        # Create padding mask (1 for valid elements, 0 for padding)
        padding_mask = (elements.sum(dim=2) != 0).float()  # [batch, seq_len-1]
        
        # Project each feature separately
        id_embeds = self.element_projection(element_ids)            # [batch, seq_len-1, 32]
        percentage_embeds = self.percentage_projection(percentages) # [batch, seq_len-1, 32]
        feature_embeds = self.feature_projection(element_features)  # [batch, seq_len-1, 64]
        
        # Combine all element embeddings
        element_embeds = torch.cat([id_embeds, percentage_embeds, feature_embeds], dim=2)  # [batch, seq_len-1, 128]
        
        # Project calculated features
        calc_features_embed = self.calc_features_projection(calc_features).unsqueeze(1)  # [batch, 1, 128]
        
        # Combine all tokens for transformer input
        transformer_input = torch.cat([element_embeds, calc_features_embed], dim=1)  # [batch, seq_len, 128]
        
        # Add calculated features token to the padding mask
        full_padding_mask = torch.cat([padding_mask, torch.ones(batch_size, 1, device=x.device)], dim=1)
        
        # Convert to transformer mask format (True for padding positions)
        attn_mask = (full_padding_mask == 0)  # [batch, seq_len]
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            transformer_input, 
            src_key_padding_mask=attn_mask
        )  # [batch, seq_len, hidden_dim]
        
        # Get calculated features representation (last token)
        calc_features_repr = transformer_output[:, -1, :]  # [batch, hidden_dim]
        
        # Apply element attention to focus on important elements
        element_repr = transformer_output[:, :-1, :]  # [batch, seq_len-1, hidden_dim]
        query = calc_features_repr.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Create element attention mask
        element_attn_mask = attn_mask[:, :-1]  # Exclude calculated features token
        
        # Apply attention
        attended_elements, _ = self.element_attention(
            query,
            element_repr,
            element_repr,
            key_padding_mask=element_attn_mask
        )  # [batch, 1, hidden_dim]
        
        attended_elements = attended_elements.squeeze(1)  # [batch, hidden_dim]
        
        # Combine calculated features and attended element representation
        combined_repr = torch.cat([calc_features_repr, attended_elements], dim=1)  # [batch, hidden_dim*2]
        fused_repr = self.fusion_layer(combined_repr)  # [batch, hidden_dim]
        
        # Final prediction
        output = self.regression_head(fused_repr)  # [batch, 1]
        
        return output

# # Instantiate model with your configuration
# model = ChemicalTransformer(
#     feature_dim=5,
#     hidden_dim=1024,
#     num_heads=512,
#     num_layers=2, 
#     dropout=0.1,
#     max_seq_length=512
# )
# summary(model)

