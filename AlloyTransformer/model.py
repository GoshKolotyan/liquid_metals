import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import logging
import numpy as np
import time
from dataloader import LM_Dataset, collate_fn

# Define logging levels beyond the standard ones
TRACE = 5  # Even more detailed than DEBUG
logging.addLevelName(TRACE, "TRACE")

def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)

logging.Logger.trace = trace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alloy_transformer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AlloyTransformer")

class LoggingLevel:
    """Constants for different logging verbosity levels"""
    MINIMAL = 1   # Basic info only (init, epoch results)
    STANDARD = 2  # Regular info (batch results, layer summaries)
    DETAILED = 3  # Detailed info (tensors, statistics, timing)
    TRACE = 4     # Maximum detail (attention matrices, all tensors)

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
        log_level=LoggingLevel.STANDARD
    ):
        super(AlloyTransformer, self).__init__()
        
        self.log_level = log_level
        self.d_model = d_model
        
        if self.log_level >= LoggingLevel.MINIMAL:
            logger.info(f"Initializing AlloyTransformer with: input_dim={input_dim}, d_model={d_model}, "
                        f"nhead={nhead}, num_layers={num_layers}, log_level={log_level}")
        
        # Initial feature embedding
        self.feature_embedding = nn.Linear(input_dim, d_model)
        
        # Learnable position/role embeddings
        self.position_embedding = nn.Parameter(torch.zeros(1, num_positions, d_model))
        nn.init.xavier_uniform_(self.position_embedding)
        
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info(f"Position embedding shape: {self.position_embedding.shape}")
        
        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )
        
        # Layer names for debugging
        self.layer_names = {
            "input": "Input",
            "embedding": "Feature Embedding",
            "position": "Position Embedding",
            "encoder": [f"Transformer Layer {i+1}" for i in range(num_layers)],
            "pooling": "Global Pooling",
            "regression": "Regression Head",
            "output": "Final Output"
        }
        
        # Hooks for debugging
        if self.log_level >= LoggingLevel.DETAILED:
            self.activation = {}
            self._register_hooks()
        
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
    
    def _register_hooks(self):
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info("Registering hooks for activation tracking")
        
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        
        # Register hooks
        self.feature_embedding.register_forward_hook(get_activation('embedding'))
    
    def _log_tensor(self, name, tensor, sample_idx=0, level=LoggingLevel.DETAILED):
        """Log tensor shape and sample values based on logging level"""
        if self.log_level < level:
            return
            
        shape_str = str(tensor.shape)
        
        # Basic shape logging for STANDARD level
        if level <= LoggingLevel.STANDARD:
            logger.info(f"{name} shape: {shape_str}")
            return
            
        # Get statistics for DETAILED level
        with torch.no_grad():
            stats = {
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "mean": tensor.mean().item(),
                "std": tensor.std().item()
            }
        
        # Get sample values for TRACE level
        if self.log_level >= LoggingLevel.TRACE:
            if tensor.dim() >= 3:  # 3D tensor (batch, seq, features)
                sample = tensor[sample_idx, 0, :5].detach().cpu().numpy()
            elif tensor.dim() == 2:  # 2D tensor (batch, features)
                sample = tensor[sample_idx, :5].detach().cpu().numpy()
            else:  # 1D tensor
                sample = tensor[:5].detach().cpu().numpy() if tensor.shape[0] >= 5 else tensor.detach().cpu().numpy()
            
            sample_str = np.array2string(sample, precision=4, suppress_small=True)
            
            logger.info(f"{name} shape: {shape_str}")
            logger.info(f"{name} stats: {stats}")
            logger.info(f"{name} sample: {sample_str}")
        else:
            # Just stats for DETAILED level
            logger.info(f"{name} shape: {shape_str}, stats: {stats}")
    
    def forward(self, x):
        batch_size, num_positions, features = x.shape
        
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info(f"\n{'='*50}\nForward pass started with input shape: {x.shape}")
        
        # Log input
        self._log_tensor(self.layer_names["input"], x, level=LoggingLevel.DETAILED)
        
        start_time = time.time()
        
        # Feature embedding
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info("\nStep 1: Projecting raw features to embedding space")
        
        x = self.feature_embedding(x)
        self._log_tensor(self.layer_names["embedding"], x, level=LoggingLevel.DETAILED)
        
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"Embedding step took {time.time() - start_time:.4f} seconds")
        
        # Position embeddings
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info("\nStep 2: Adding position/role embeddings")
        
        pos_emb_time = time.time()
        position_emb = self.position_embedding.expand(batch_size, -1, -1)
        self._log_tensor("Position Embedding", position_emb, level=LoggingLevel.DETAILED)
        
        x = x + position_emb
        self._log_tensor("Embedding + Position", x, level=LoggingLevel.DETAILED)
        
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"Position embedding took {time.time() - pos_emb_time:.4f} seconds")
        
        # Transformer encoder
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info("\nStep 3: Processing through transformer encoder layers")
        
        encoder_time = time.time()
        
        # Log attention patterns if in TRACE mode
        if self.log_level >= LoggingLevel.TRACE:
            logger.trace("Attention visualization would be shown here in trace mode")
        
        transformer_output = self.transformer_encoder(x)
        self._log_tensor("Transformer Output", transformer_output, level=LoggingLevel.DETAILED)
        
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"Transformer encoding took {time.time() - encoder_time:.4f} seconds")
        
        # Global pooling
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info("\nStep 4: Global pooling over token dimension")
        
        pooling_time = time.time()
        pooled = torch.mean(transformer_output, dim=1)
        self._log_tensor(self.layer_names["pooling"], pooled, level=LoggingLevel.DETAILED)
        
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"Pooling step took {time.time() - pooling_time:.4f} seconds")
        
        # Regression head
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info("\nStep 5: Prediction through regression head")
        
        regression_time = time.time()
        melting_point = self.regression_head(pooled).squeeze(-1)
        self._log_tensor(self.layer_names["output"], melting_point, level=LoggingLevel.DETAILED)
        
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"Regression step took {time.time() - regression_time:.4f} seconds")
        
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info(f"\nTotal forward pass took {time.time() - start_time:.4f} seconds")
            logger.info(f"{'='*50}\n")
        
        return melting_point


class PropertyFocusedAttention(nn.Module):
    """Custom attention implementation with detailed logging"""
    def __init__(self, d_model, num_heads, dropout=0.1, property_bias=None, log_level=LoggingLevel.STANDARD):
        super(PropertyFocusedAttention, self).__init__()
        
        self.log_level = log_level
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info(f"Initializing PropertyFocusedAttention with: d_model={d_model}, "
                      f"num_heads={num_heads}, head_dim={self.head_dim}")
        
        # Check if dimensions are compatible
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Projection matrices
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Property bias - can be used to bias attention towards specific properties
        if property_bias is not None:
            if self.log_level >= LoggingLevel.STANDARD:
                logger.info("Initializing property bias with melting point focus")
            
            # Initialize with a non-zero tensor that won't cause in-place operation issues
            bias_init = torch.zeros(num_heads, 5, 5)
            # Initialize one head to focus on melting point (last feature)
            bias_init[0, :, -1] = 1.0
            self.property_bias = nn.Parameter(bias_init)
            
            if self.log_level >= LoggingLevel.DETAILED:
                logger.info(f"Property bias shape: {self.property_bias.shape}")
                
                if self.log_level >= LoggingLevel.TRACE:
                    logger.trace(f"Property bias sample:\n{self.property_bias[0, :, :].detach().cpu().numpy()}")
        else:
            self.property_bias = None
            
        self.dropout = nn.Dropout(dropout)
        
    def _log_attention(self, name, tensor, batch_idx=0, head_idx=0):
        """Log attention patterns"""
        # Only log attention matrices in TRACE mode
        if self.log_level < LoggingLevel.TRACE:
            return
            
        logger.trace(f"{name} shape: {tensor.shape}")
        
        # For attention weights, show them as a matrix for one head
        if tensor.dim() == 4:  # [batch, heads, seq, seq]
            attn_matrix = tensor[batch_idx, head_idx].detach().cpu().numpy()
            logger.trace(f"{name} for batch {batch_idx}, head {head_idx}:")
            
            # Format as a readable matrix
            matrix_str = np.array2string(attn_matrix, precision=3, suppress_small=True)
            logger.trace(f"{matrix_str}")
        
    def forward(self, x):
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info("\n--- PropertyFocusedAttention Forward ---")
            logger.info(f"Input shape: {x.shape}")
        
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        if self.log_level >= LoggingLevel.TRACE:
            logger.trace(f"Q projection shape: {q.shape}")
            logger.trace(f"K projection shape: {k.shape}")
            logger.trace(f"V projection shape: {v.shape}")
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.log_level >= LoggingLevel.TRACE:
            logger.trace(f"Q after reshape: {q.shape} [batch, heads, seq, head_dim]")
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if self.log_level >= LoggingLevel.TRACE:
            logger.trace(f"Raw attention scores shape: {scores.shape}")
            self._log_attention("Raw attention scores", scores)
        
        # Apply property bias if specified
        if self.property_bias is not None:
            bias = self.property_bias.unsqueeze(0)
            scores = scores + bias
            
            if self.log_level >= LoggingLevel.TRACE:
                logger.trace("Added property bias to attention scores")
                self._log_attention("Biased attention scores", scores)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        
        if self.log_level >= LoggingLevel.DETAILED:
            self._log_attention("Attention weights after softmax", attn_weights)
            
            # Log feature importance
            if self.log_level >= LoggingLevel.TRACE:
                avg_attn = attn_weights.mean(dim=0).mean(dim=0)  # Average over batch and heads
                logger.trace("Average attention per position:")
                logger.trace(np.array2string(avg_attn.detach().cpu().numpy(), precision=3))
        
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        if self.log_level >= LoggingLevel.TRACE:
            logger.trace(f"Output after attention shape: {output.shape}")
        
        # Transpose back and reshape
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.out_proj(output)
        
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"Final output shape: {output.shape}")
            logger.info("--- PropertyFocusedAttention Forward End ---\n")
        
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
        log_level=LoggingLevel.STANDARD
    ):
        super(EnhancedAlloyTransformer, self).__init__()
        
        self.log_level = log_level
        self.d_model = d_model
        self.num_layers = num_layers
        
        if self.log_level >= LoggingLevel.MINIMAL:
            logger.info(f"Initializing EnhancedAlloyTransformer with: input_dim={input_dim}, "
                      f"d_model={d_model}, nhead={nhead}, num_layers={num_layers}, "
                      f"use_property_focus={use_property_focus}, log_level={log_level}")
        
        # Initial feature embedding
        self.feature_embedding = nn.Linear(input_dim, d_model)
        
        # Learnable position/role embeddings
        self.position_embedding = nn.Parameter(torch.zeros(1, num_positions, d_model))
        nn.init.xavier_uniform_(self.position_embedding)
        
        # Custom encoder layers with property-focused attention
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if self.log_level >= LoggingLevel.STANDARD:
                logger.info(f"Creating layer {i+1}/{num_layers}")
            
            # Attention mechanism with property focus
            if use_property_focus:
                attention = PropertyFocusedAttention(
                    d_model=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    property_bias=True,
                    log_level=log_level
                )
                if self.log_level >= LoggingLevel.STANDARD:
                    logger.info(f"Layer {i+1}: Using PropertyFocusedAttention")
            else:
                attention = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True
                )
                if self.log_level >= LoggingLevel.STANDARD:
                    logger.info(f"Layer {i+1}: Using standard MultiheadAttention")
            
            # Add the full encoder layer
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(d_model),
                attention,
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model)
                ),
                nn.Dropout(dropout)
            ]))
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        if self.log_level >= LoggingLevel.MINIMAL:
            logger.info("Enhanced model initialization complete")
    
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
            
        shape_str = str(tensor.shape)
        
        # Basic shape logging for STANDARD level
        if level <= LoggingLevel.STANDARD:
            logger.info(f"{name} shape: {shape_str}")
            return
            
        # Get statistics for DETAILED level
        with torch.no_grad():
            stats = {
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "mean": tensor.mean().item(),
                "std": tensor.std().item()
            }
        
        # Get sample values for TRACE level
        if self.log_level >= LoggingLevel.TRACE:
            if tensor.dim() >= 3:  # 3D tensor (batch, seq, features)
                sample = tensor[sample_idx, 0, :5].detach().cpu().numpy()
            elif tensor.dim() == 2:  # 2D tensor (batch, features)
                sample = tensor[sample_idx, :5].detach().cpu().numpy()
            else:  # 1D tensor
                sample = tensor[:5].detach().cpu().numpy() if tensor.shape[0] >= 5 else tensor.detach().cpu().numpy()
            
            sample_str = np.array2string(sample, precision=4, suppress_small=True)
            
            logger.info(f"{name} shape: {shape_str}")
            logger.info(f"{name} stats: {stats}")
            logger.info(f"{name} sample: {sample_str}")
        else:
            # Just stats for DETAILED level
            logger.info(f"{name} shape: {shape_str}, stats: {stats}")
    
    def forward(self, x):
        start_time = time.time()
        
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info(f"\n{'='*50}\nEnhanced model forward pass with input shape: {x.shape}")
        
        # Project features to embedding dimension
        embed_time = time.time()
        
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info("\nStep 1: Feature embedding")
            
        x = self.feature_embedding(x)
        self._log_tensor("Feature embedding output", x, level=LoggingLevel.DETAILED)
        
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"Feature embedding took {time.time() - embed_time:.4f} seconds")
        
        # Add position/role embeddings
        pos_time = time.time()
        
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info("\nStep 2: Position embedding")
            
        batch_size = x.shape[0]
        position_emb = self.position_embedding.expand(batch_size, -1, -1)
        self._log_tensor("Position embedding", position_emb, level=LoggingLevel.DETAILED)
        
        x = x + position_emb
        self._log_tensor("Combined embeddings", x, level=LoggingLevel.DETAILED)
        
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"Position embedding took {time.time() - pos_time:.4f} seconds")
        
        # Apply transformer encoder layers manually for more control
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info(f"\nStep 3: Processing through {self.num_layers} transformer layers")
        
        for i, (layer_norm1, attn, dropout1, layer_norm2, ffn, dropout2) in enumerate(self.layers):
            layer_time = time.time()
            
            if self.log_level >= LoggingLevel.STANDARD:
                logger.info(f"\n  Layer {i+1}/{self.num_layers}:")
            
            # Pre-norm architecture
            if self.log_level >= LoggingLevel.DETAILED:
                logger.info(f"  3.{i+1}.1: Layer normalization 1")
                
            norm_x = layer_norm1(x)
            self._log_tensor(f"Layer {i+1} norm1 output", norm_x, level=LoggingLevel.TRACE)
            
            # Apply attention
            if self.log_level >= LoggingLevel.DETAILED:
                logger.info(f"  3.{i+1}.2: Self-attention")
                
            attn_time = time.time()
            if isinstance(attn, PropertyFocusedAttention):
                attn_output = attn(norm_x)
            else:
                attn_output, _ = attn(norm_x, norm_x, norm_x)
            
            self._log_tensor(f"Layer {i+1} attention output", attn_output, level=LoggingLevel.TRACE)
            
            if self.log_level >= LoggingLevel.DETAILED:
                logger.info(f"  Attention computation took {time.time() - attn_time:.4f} seconds")
            
            # Residual connection and dropout
            if self.log_level >= LoggingLevel.DETAILED:
                logger.info(f"  3.{i+1}.3: Residual connection + dropout")
                
            x = x + dropout1(attn_output)
            self._log_tensor(f"Layer {i+1} after residual", x, level=LoggingLevel.TRACE)
            
            # FFN with pre-norm
            if self.log_level >= LoggingLevel.DETAILED:
                logger.info(f"  3.{i+1}.4: Layer normalization 2")
                
            norm_x = layer_norm2(x)
            
            if self.log_level >= LoggingLevel.DETAILED:
                logger.info(f"  3.{i+1}.5: Feed-forward network")
                
            ffn_time = time.time()
            ffn_output = ffn(norm_x)
            self._log_tensor(f"Layer {i+1} FFN output", ffn_output, level=LoggingLevel.TRACE)
            
            if self.log_level >= LoggingLevel.DETAILED:
                logger.info(f"  FFN computation took {time.time() - ffn_time:.4f} seconds")
            
            # Residual connection and dropout
            if self.log_level >= LoggingLevel.DETAILED:
                logger.info(f"  3.{i+1}.6: Residual connection + dropout")
                
            x = x + dropout2(ffn_output)
            self._log_tensor(f"Layer {i+1} final output", x, level=LoggingLevel.TRACE)
            
            if self.log_level >= LoggingLevel.DETAILED:
                logger.info(f"  Layer {i+1} total processing took {time.time() - layer_time:.4f} seconds")
        
        # Final layer norm
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info("\nStep 4: Final layer normalization")
            
        norm_time = time.time()
        x = self.final_norm(x)
        self._log_tensor("Final normalized output", x, level=LoggingLevel.DETAILED)
        
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"Final normalization took {time.time() - norm_time:.4f} seconds")
        
        # Global average pooling over positions
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info("\nStep 5: Global pooling")
            
        pool_time = time.time()
        pooled = torch.mean(x, dim=1)
        self._log_tensor("Pooled output", pooled, level=LoggingLevel.DETAILED)
        
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"Pooling took {time.time() - pool_time:.4f} seconds")
        
        # Predict melting point
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info("\nStep 6: Regression prediction")
            
        reg_time = time.time()
        melting_point = self.regression_head(pooled).squeeze(-1)
        self._log_tensor("Predicted melting point", melting_point, level=LoggingLevel.DETAILED)
        
        if self.log_level >= LoggingLevel.DETAILED:
            logger.info(f"Regression took {time.time() - reg_time:.4f} seconds")
        
        if self.log_level >= LoggingLevel.STANDARD:
            logger.info(f"\nTotal forward pass took {time.time() - start_time:.4f} seconds")
            logger.info(f"{'='*50}\n")
        
        return melting_point


# Training function with configurable logging
def train_model(model, train_loader, valid_loader=None, epochs=10, 
                learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu',
                log_level=LoggingLevel.STANDARD):
    """Train the model with configurable logging verbosity"""
    if log_level >= LoggingLevel.MINIMAL:
        logger.info(f"Starting training on device: {device}")
        logger.info(f"Training for {epochs} epochs with learning rate: {learning_rate}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # For tracking best model
    best_valid_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        if log_level >= LoggingLevel.MINIMAL:
            logger.info(f"\n{'*'*20} Epoch {epoch+1}/{epochs} {'*'*20}")
        
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, (composition, target) in enumerate(train_loader):
            batch_start = time.time()
            
            if log_level >= LoggingLevel.STANDARD:
                logger.info(f"\nBatch {batch_idx+1}/{len(train_loader)}")
            # Move data to device
            composition = composition.to(device)
            target = target.to(device)
           
            if log_level >= LoggingLevel.DETAILED:
               logger.info(f"Input shape: {composition.shape}, Target shape: {target.shape}")
           
           # Forward pass
            optimizer.zero_grad()
            output = model(composition)
           
           # Calculate loss
            loss = criterion(output, target)
            train_losses.append(loss.item())
           
           # Backward pass and optimize
            loss.backward()
            optimizer.step()
           
           # Log batch details
            if (batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1) and log_level >= LoggingLevel.STANDARD:
               logger.info(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
               
               # Log sample predictions in DETAILED mode
               if log_level >= LoggingLevel.DETAILED and composition.shape[0] > 0:
                   with torch.no_grad():
                       for i in range(min(3, composition.shape[0])):
                           logger.info(f"Sample {i+1}: Predicted={output[i].item():.2f}, Actual={target[i].item():.2f}")
           
            if log_level >= LoggingLevel.DETAILED:
               logger.info(f"Batch processing took {time.time() - batch_start:.4f} seconds")
       
       # Validation phase
        if valid_loader is not None:
           valid_loss = validate_model(model, valid_loader, criterion, device, log_level)
           
           if log_level >= LoggingLevel.MINIMAL:
               logger.info(f"Validation Loss: {valid_loss:.4f}")
           
           # Save best model
           if valid_loss < best_valid_loss:
               best_valid_loss = valid_loss
               torch.save(model.state_dict(), 'best_alloy_model.pt')
               
               if log_level >= LoggingLevel.MINIMAL:
                   logger.info("Saved new best model!")
       
       # Log epoch summary
    avg_train_loss = sum(train_losses) / len(train_losses)
    
    if log_level >= LoggingLevel.MINIMAL:
        logger.info(f"Epoch {epoch+1}/{epochs} completed in {time.time() - epoch_start:.2f} seconds")
        logger.info(f"Average Training Loss: {avg_train_loss:.4f}")

    if log_level >= LoggingLevel.MINIMAL:
        logger.info("Training completed!")
        
    return model


def validate_model(model, valid_loader, criterion, device, log_level=LoggingLevel.STANDARD):
   """Validate the model with configurable logging verbosity"""
   model.eval()
   valid_losses = []
   
   if log_level >= LoggingLevel.STANDARD:
       logger.info("Starting validation...")
       
   with torch.no_grad():
       for batch_idx, (composition, target) in enumerate(valid_loader):
           composition = composition.to(device)
           target = target.to(device)
           
           output = model(composition)
           loss = criterion(output, target)
           valid_losses.append(loss.item())
           
           if batch_idx % 10 == 0 and log_level >= LoggingLevel.DETAILED:
               logger.info(f"Validation Batch {batch_idx+1}/{len(valid_loader)}, Loss: {loss.item():.4f}")
   
   avg_valid_loss = sum(valid_losses) / len(valid_losses)
   return avg_valid_loss


def set_logging_level(level):
   """Set the logging level for the logger"""
   if level == LoggingLevel.MINIMAL:
       logger.setLevel(logging.INFO)
   elif level == LoggingLevel.STANDARD:
       logger.setLevel(logging.INFO)
   elif level == LoggingLevel.DETAILED:
       logger.setLevel(logging.DEBUG)
   elif level == LoggingLevel.TRACE:
       logger.setLevel(TRACE)
   else:
       logger.setLevel(logging.INFO)


# Example usage function
def example_integration_with_dataloader(log_level=LoggingLevel.STANDARD):
   """Run a full example with your data loader and configurable logging level"""
   import torch
   from torch.utils.data import DataLoader
   
   # Set logging level
   set_logging_level(log_level)
   
   if log_level >= LoggingLevel.MINIMAL:
       logger.info(f"Starting AlloyTransformer example with log level: {log_level}")
   
   # Load data
   if log_level >= LoggingLevel.MINIMAL:
       logger.info("Loading datasets")
       
   try:
       train_dataset = LM_Dataset(data_path="Data/Component_Stratified_Split/train.csv")
       valid_dataset = LM_Dataset(data_path="Data/Component_Stratified_Split/valid.csv")
       
       train_loader = DataLoader(dataset=train_dataset, batch_size=64, collate_fn=collate_fn)
       valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, collate_fn=collate_fn)
       
       if log_level >= LoggingLevel.MINIMAL:
           logger.info(f"Loaded training set with {len(train_dataset)} samples")
           logger.info(f"Loaded validation set with {len(valid_dataset)} samples")
       
       # Sample a batch to determine input shape
       for sample_batch in train_loader:
           composition_tensor, target_tensor = sample_batch
           
           if log_level >= LoggingLevel.STANDARD:
               logger.info(f"Sample batch composition shape: {composition_tensor.shape}")
               logger.info(f"Sample batch target shape: {target_tensor.shape}")
               
           break
       
       # Create model based on input shape
       input_dim = composition_tensor.shape[2]  # Feature dimension
       num_positions = composition_tensor.shape[1]  # Sequence length
       
       if log_level >= LoggingLevel.MINIMAL:
           logger.info(f"Creating model with input_dim={input_dim}, num_positions={num_positions}")
       
       model = EnhancedAlloyTransformer(
           input_dim=input_dim,
           d_model=64,
           nhead=4,
           num_layers=3,
           dim_feedforward=128,
           dropout=0.1,
           num_positions=num_positions,
           use_property_focus=True,
           log_level=log_level
       )
       
       # Train the model
       if log_level >= LoggingLevel.MINIMAL:
           logger.info("Starting model training")
           
       train_model(
           model=model,
           train_loader=train_loader,
           valid_loader=valid_loader,
           epochs=120,
           learning_rate=0.001,
           log_level=log_level
       )
       
       if log_level >= LoggingLevel.MINIMAL:
           logger.info("Example completed successfully")
           
       return model
       
   except Exception as e:
       logger.error(f"Error in example integration: {str(e)}")
       import traceback
       logger.error(traceback.format_exc())


if __name__ == "__main__":
   # Choose the logging level here
   # LoggingLevel.MINIMAL - Basic info only (init, epoch results)
   # LoggingLevel.STANDARD - Regular info (batch results, layer summaries)
   # LoggingLevel.DETAILED - Detailed info (tensors, statistics, timing)
   # LoggingLevel.TRACE - Maximum detail (attention matrices, all tensors)
   
   import argparse
   
   parser = argparse.ArgumentParser(description='Train the AlloyTransformer model with configurable logging')
   parser.add_argument('--log_level', type=int, default=1, help='Logging verbosity (1=Minimal, 2=Standard, 3=Detailed, 4=Trace)')
   parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
   parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs')
   parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
   parser.add_argument('--d_model', type=int, default=64, help='Model embedding dimension')
   parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
   parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
   
   args = parser.parse_args()
   
   # Convert argument to LoggingLevel
   log_level = args.log_level
   if log_level < 1 or log_level > 4:
       print(f"Invalid log level: {log_level}. Using default (STANDARD).")
       log_level = LoggingLevel.STANDARD
   else:
       log_level = log_level  # 1-4 maps directly to our LoggingLevel values
   
   # Run with specified logging level
   example_integration_with_dataloader(log_level=log_level)     