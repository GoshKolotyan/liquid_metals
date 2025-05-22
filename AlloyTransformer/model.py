import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import logging
import numpy as np
import time
from dataloader import LM_Dataset, collate_fn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("alloy_transformer.log"), logging.StreamHandler()],
)
logger = logging.getLogger("AlloyTransformer")


class LoggingLevel:
    """Constants for different logging verbosity levels"""
    MINIMAL = 1  # Basic info only (init, epoch results)
    STANDARD = 2  # Regular info (batch results, layer summaries)
    DETAILED = 3  # Detailed info (tensors, statistics, timing)
    TRACE = 4  # Maximum detail (attention matrices, all tensors)


class AlloyTransformer(nn.Module):
    def __init__(
        self,
        input_dim=5,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        num_positions=5,
        log_level=LoggingLevel.MINIMAL,
    ):
        super(AlloyTransformer, self).__init__()

        self.log_level = log_level
        self.d_model = d_model

        if self.log_level >= LoggingLevel.MINIMAL:
            logger.info(
                f"Initializing AlloyTransformer with: input_dim={input_dim}, d_model={d_model}, "
                f"nhead={nhead}, num_layers={num_layers}"
            )

        # Initial feature embedding
        self.feature_embedding = nn.Linear(input_dim, d_model)

        # Learnable position/role embeddings
        self.position_embedding = nn.Parameter(torch.zeros(1, num_positions, d_model))
        nn.init.xavier_uniform_(self.position_embedding)

        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layers, num_layers=num_layers
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

        # Layer names for debugging
        self.layer_names = {
            "input": "Input",
            "embedding": "Feature Embedding",
            "position": "Position Embedding",
            "encoder": [f"Transformer Layer {i+1}" for i in range(num_layers)],
            "pooling": "Global Pooling",
            "regression": "Regression Head",
            "output": "Final Output",
        }

        # Initialize weights
        self._init_weights()

        if self.log_level >= LoggingLevel.MINIMAL:
            logger.info("Model initialized successfully")

    def _init_weights(self):
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info("Initializing weights with Xavier uniform")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _log_tensor(self, name, tensor, sample_idx=0, level=LoggingLevel.DETAILED):
        """Log tensor shape and sample values based on logging level"""
        if self.log_level < level:
            return

        # Only log shape at DETAILED level
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"{name} shape: {tensor.shape}")

    def forward(self, x):
        batch_size, num_positions, features = x.shape

        if self.log_level >= LoggingLevel.STANDARD:
            logger.info(f"Forward pass started with input shape: {x.shape}")

        start_time = time.time()

        # Feature embedding
        x = self.feature_embedding(x)
        
        # Position embeddings
        position_emb = self.position_embedding.expand(batch_size, -1, -1)
        x = x + position_emb

        # Transformer encoder
        transformer_output = self.transformer_encoder(x)

        # Global pooling
        pooled = torch.mean(transformer_output, dim=1)

        # Regression head
        melting_point = self.regression_head(pooled).squeeze(-1)

        if self.log_level >= LoggingLevel.MINIMAL:
            logger.info(f"Forward pass completed in {time.time() - start_time:.4f} seconds")

        return melting_point


class PropertyFocusedAttention(nn.Module):
    """Custom attention implementation with detailed logging"""

    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.1,
        property_bias=None,
        log_level=LoggingLevel.MINIMAL,
    ):
        super(PropertyFocusedAttention, self).__init__()

        self.log_level = log_level
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        if self.log_level >= LoggingLevel.STANDARD:
            logger.info(
                f"Initializing PropertyFocusedAttention with: d_model={d_model}, "
                f"num_heads={num_heads}, head_dim={self.head_dim}"
            )

        # Check if dimensions are compatible
        assert (
            self.head_dim * num_heads == d_model
        ), "d_model must be divisible by num_heads"

        # Projection matrices
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Property bias - can be used to bias attention towards specific properties
        if property_bias is not None:
            # Initialize with a non-zero tensor that won't cause in-place operation issues
            bias_init = torch.zeros(num_heads, 5, 5)
            # Initialize one head to focus on melting point (last feature)
            bias_init[0, :, -1] = 1.0
            self.property_bias = nn.Parameter(bias_init)
        else:
            self.property_bias = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"PropertyFocusedAttention input shape: {x.shape}")

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
            bias = self.property_bias.unsqueeze(0)
            scores = scores + bias

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Transpose back and reshape
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        # Final projection
        output = self.out_proj(output)

        return output


class EnhancedAlloyTransformer(nn.Module):
    """Enhanced model with flexible logging levels"""

    def __init__(
        self,
        input_dim=5,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        num_positions=5,
        use_property_focus=True,
        log_level=LoggingLevel.MINIMAL,
    ):
        super(EnhancedAlloyTransformer, self).__init__()

        self.log_level = log_level
        self.d_model = d_model
        self.num_layers = num_layers

        if self.log_level >= LoggingLevel.MINIMAL:
            logger.info(
                f"Initializing EnhancedAlloyTransformer with: input_dim={input_dim}, "
                f"d_model={d_model}, nhead={nhead}, num_layers={num_layers}, "
                f"use_property_focus={use_property_focus}"
            )

        # Initial feature embedding
        self.feature_embedding = nn.Linear(input_dim, d_model)

        # Learnable position/role embeddings
        self.position_embedding = nn.Parameter(torch.zeros(1, num_positions, d_model))
        nn.init.xavier_uniform_(self.position_embedding)

        # Custom encoder layers with property-focused attention
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Attention mechanism with property focus
            if use_property_focus:
                attention = PropertyFocusedAttention(
                    d_model=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    property_bias=True,
                    log_level=log_level,
                )
            else:
                attention = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True,
                )

            # Add the full encoder layer
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(d_model),
                        attention,
                        nn.Dropout(dropout),
                        nn.LayerNorm(d_model),
                        nn.Sequential(
                            nn.Linear(d_model, dim_feedforward),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim_feedforward, d_model),
                        ),
                        nn.Dropout(dropout),
                    ]
                )
            )

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

        # Initialize weights
        self._init_weights()

        if self.log_level >= LoggingLevel.MINIMAL:
            logger.info("Enhanced model initialization complete")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _log_tensor(self, name, tensor, sample_idx=0, level=LoggingLevel.DETAILED):
        """Log tensor shape and sample values based on logging level"""
        if self.log_level < level:
            return

        # Only log shape at DETAILED level
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"{name} shape: {tensor.shape}")

    def forward(self, x):
        start_time = time.time()

        if self.log_level >= LoggingLevel.STANDARD:
            logger.info(f"Enhanced model forward pass with input shape: {x.shape}")

        # Project features to embedding dimension
        x = self.feature_embedding(x)

        # Add position/role embeddings
        batch_size = x.shape[0]
        position_emb = self.position_embedding.expand(batch_size, -1, -1)
        x = x + position_emb

        # Apply transformer encoder layers manually for more control
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info(f"Processing through {self.num_layers} transformer layers")

        for i, (layer_norm1, attn, dropout1, layer_norm2, ffn, dropout2) in enumerate(
            self.layers
        ):
            layer_time = time.time()

            # Pre-norm architecture
            norm_x = layer_norm1(x)

            # Apply attention
            if isinstance(attn, PropertyFocusedAttention):
                attn_output = attn(norm_x)
            else:
                attn_output, _ = attn(norm_x, norm_x, norm_x)

            # Residual connection and dropout
            x = x + dropout1(attn_output)

            # FFN with pre-norm
            norm_x = layer_norm2(x)
            ffn_output = ffn(norm_x)

            # Residual connection and dropout
            x = x + dropout2(ffn_output)

            if self.log_level >= LoggingLevel.DETAILED:
                logger.info(f"Layer {i+1} processed in {time.time() - layer_time:.4f} seconds")

        # Final layer norm
        x = self.final_norm(x)

        # Global average pooling over positions
        pooled = torch.mean(x, dim=1)

        # Predict melting point
        melting_point = self.regression_head(pooled).squeeze(-1)

        if self.log_level >= LoggingLevel.MINIMAL:
            logger.info(f"Total forward pass completed in {time.time() - start_time:.4f} seconds")

        return melting_point


if __name__ == "__main__":
    # Example usage
    model = EnhancedAlloyTransformer(
        input_dim=5,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        num_positions=5,
        log_level=LoggingLevel.DETAILED,
    )

    # Dummy input
    x = torch.randn(
        32, 5, 5
    )  # Batch size of 32, sequence length of 5, feature dimension of 5

    # Forward pass
    output = model(x)

    print("Output shape:", output.shape)
