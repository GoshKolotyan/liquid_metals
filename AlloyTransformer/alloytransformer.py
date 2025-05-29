import torch
from torch.nn.init import xavier_uniform_, zeros_, normal_
from torch.nn.functional import softmax
from torch.nn import (
    Module,
    Linear,
    ModuleList,
    AdaptiveAvgPool1d,
    Dropout,
    LayerNorm,
    MultiheadAttention,
    Conv1d,
    BatchNorm1d,
    Sequential,
    ReLU,
    Parameter,

)


class PropertyFocusedAttention(Module):
    def __init__(self, 
                 d_model:int, 
                 num_heads:int, 
                 dropout:float, 
                 property_bias:bool):
        super(PropertyFocusedAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        #projection matrices
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

        # Property bias - can be used to bias attention towards specific properties
        if property_bias is not None:
            # Initialize with a non-zero tensor that won't cause in-place operation issues
            bias_init = torch.zeros(num_heads, 5, 5)
            # Initialize one head to focus on melting point (last feature)
            bias_init[0, :, -1] = 1.0
            self.property_bias = Parameter(bias_init)
        else:
            self.property_bias = None

        self.dropout = Dropout(dropout)
    
    def forward(self, x:torch.Tensor)-> torch.Tensor:

        batch_size, seq_len, _ = x.shape

        # Project queries, keys, values
        q:torch.Tensor = self.q_proj(x)
        k:torch.Tensor = self.k_proj(x)
        v:torch.Tensor = self.v_proj(x)

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
        attn_weights = softmax(scores, dim=-1)
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
class CNNRegressionHead(Module):

    def __init__(self, d_model:int, num_conv_layers:int, dropout:float, use_batch_norm:bool):
        super(CNNRegressionHead, self).__init__()
        self.d_model = d_model
        self.use_batch_norm = use_batch_norm
        
        conv_layers = []
        in_channels = d_model

        for i in range(num_conv_layers):
            # Reduce channels progressively
            out_channels = max(32, d_model // (2 ** (i + 1)))
            
            conv_layers.append(Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not use_batch_norm
            ))
            
            if use_batch_norm:
                conv_layers.append(BatchNorm1d(out_channels))
            
            conv_layers.append(ReLU(inplace=True))
            conv_layers.append(Dropout(dropout))
            
            in_channels = out_channels
        
        self.conv_layers = Sequential(*conv_layers)

        # Global pooling options (deterministic-friendly)
        self.global_avg_pool = AdaptiveAvgPool1d(1)
        
        final_features = in_channels  # Only avg pooling
        self.final_layers = Sequential(
            Linear(final_features, final_features // 2),
            ReLU(),
            Dropout(dropout),
            Linear(final_features // 2, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(50 * "==")
        # print(18*"==","Using CNNRegressionHead",18*"==")
        # print(50 * "==")        
        # print(f"Input shape to CNNRegressionHead: {x.shape}")
        x = x.transpose(2, 1)
        
        x = self.conv_layers(x)
        # print(50 * "==")
        # print(f"Shape after convolution layers: {x.shape}")
        avg_pooled = self.global_avg_pool(x).squeeze(-1)  # (batch_size, channels)
        # print(50 * "==")
        # print(f"Shape after global average pooling: {avg_pooled.shape}")
        pooled = avg_pooled
        # print(50 * "==")
        # print(f"Shape before final layers: {pooled.shape}")
        output = self.final_layers(pooled)
        # print(50 * "==")
        # print(f"Output shape after final layers: {output.shape}")
        return output

class AlloyTransformer(Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        num_head: int,
        num_transformer_layers: int,
        num_regression_head_layers: int,
        dropout: float,
        num_positions: int,
        dim_feedforward: int,
        use_property_focus: bool,
        use_cnn_regression: bool,
    ):
        super(AlloyTransformer, self).__init__()
        self.d_model = d_model
        self.use_cnn_regression = use_cnn_regression

        # input dim is (batch_size, 5, 5)->(batch_size, 5, d_model)
        self.feature_embeddings = Linear(feature_dim, d_model)

        # learnabel role embeddings (batch_size, 5, d_model)
        self.role_embeddings = Parameter(torch.zeros(1, num_positions, d_model))
        xavier_uniform_(self.role_embeddings)

        self.layers = ModuleList()
        for _ in range(num_transformer_layers):
            if use_property_focus:
                attention = PropertyFocusedAttention(
                    d_model=d_model,
                    num_heads=num_head,
                    dropout=dropout,
                    property_bias=True,
                )
            else:
                attention = MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=num_head,
                    dropout=dropout,
                    batch_first=True,
                )

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

        # regression head
        if use_cnn_regression:
            self.regression_head = CNNRegressionHead(
                d_model=d_model,
                num_conv_layers=num_regression_head_layers,
                dropout=dropout,
                use_batch_norm=True
            )
        else:
            regression_layers = []
            for _ in range(num_regression_head_layers-1):
                    regression_layers.extend([
                    Linear(d_model, d_model),
                    ReLU(),
                    Dropout(dropout)
                ])

            regression_layers.append(Linear(d_model, 1))

            self.regression_head = Sequential(*regression_layers)

        # init wights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                normal_(m.weight)
                # xavier_uniform_(m.weight)
                # if m.bias is not None:
                #     zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x shape is (batch_size, 5, 5)
        x = self.feature_embeddings(x)
        # print("x shape after feature embedding:", x.shape)
        # print(50 * "==")
        # print(f"x shape after feature embedding: {x.shape}")
        
        # role embeddings
        batch_size = x.shape[0]
        role_embeddings = self.role_embeddings.expand(batch_size, -1, -1)
        # print(50 * "==")
        # print(f"role_embeddings shape: {role_embeddings.shape}")

        x = x + role_embeddings
        # print(50 * "==")
        # print(f"x shape after adding role embeddings: {x.shape}")

        for i, (layer_norm1, attn, dropout1, layer_norm2, ffn, dropout2) in enumerate(self.layers):
            # pre norm
            # print(50 * "==")
            norm_x = layer_norm1(x)
            # print(f"Layer {i+1} input shape: {norm_x.shape}")
            # attention
            if isinstance(attn, PropertyFocusedAttention):
                # print(50 * "==")
                # print(18*"==",f"Using PropertyFocusedAttention", 16*"==")
                # print(50 * "==")

                x = attn(norm_x)
                # print(f"Layer {i+1} attention output shape: {x.shape}")
            else:
                # print(50 * "==")
                # print(18*"==",f"Using MultiheadAttention", 19*"==")
                # print(50 * "==")                
                x, _ = attn(norm_x, norm_x, norm_x)
                # print(f"Layer {i+1} attention output shape: {x.shape}")
            
            #residual connection
            x = x + dropout1(norm_x)
            # print(50 * "==")
            # print(f"Layer {i+1} after residual connection shape: {x.shape}")
            
            #ffn with pre-norm
            norm_x = layer_norm2(x)
            # print(50 * "==")
            # print(f"Layer {i+1} ffn pre-norm shape: {norm_x.shape}")
            ffn_output = ffn(norm_x)
            # print(50 * "==")
            # print(f"Layer {i+1} ffn output shape: {ffn_output.shape}")
            
            #residual connection
            x = x + dropout2(ffn_output)
            # print(50 * "==")
            # print(f"Layer {i+1} output shape: {x.shape}")

        # final layer norm
        x = self.final_norm(x)
        # print(50 * "==")
        # print(f"Final output shape: {x.shape}")

        # global average pooling over positions 
        pooled = torch.mean(x, dim=1)
        # print(50 * "==")
        # print(f"Pooled output shape: {pooled.shape}")

        #predediction metling point
        if self.use_cnn_regression:
            melting_point = self.regression_head(x).squeeze(-1)
        else:
            melting_point = self.regression_head(pooled).squeeze(-1)

        return melting_point


if __name__ == "__main__":
    from dataloader import LM_Dataset, collate_fn
    from torch.utils.data import DataLoader
    from torchinfo import summary

    dataloader = DataLoader(
        dataset=LM_Dataset("./Data/example.csv"), collate_fn=collate_fn, batch_size=1
    )
    model = AlloyTransformer(
        feature_dim=5,
        d_model=1024,
        num_head=16,
        num_transformer_layers=5,
        num_regression_head_layers=4,
        dropout=0.1,
        num_positions=5,
        dim_feedforward=512,
        use_property_focus=True,
        use_cnn_regression=True
    )

    for batch in dataloader:
        features, target = batch
        print(f'features shape: {features.shape}')
        print(f'target shape: {target.shape}')

        output = model(features)
        # print(50 * "==")
        # print(f"output is: {output.item()}")
        break

    print(23*"==","Model",23*"==")
    # summary(model, verbose=2)