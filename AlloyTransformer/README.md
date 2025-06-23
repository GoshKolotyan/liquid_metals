## Model Architecture

```
Input Tensor [batch_size, 5, 5]    
     │    
     ▼    
┌────────────────────┐   
│  Linear Embedding  │  Transform raw features to embedding dimension      
│  [5 → d_model]     │        
└────────────────────┘        
     │    
     ▼    
┌────────────────────┐        
│ Learnable Position │  Add role-based position information 
│     Embeddings     │  for elements vs. mixture properties 
└────────────────────┘   
     │    
     ▼   ┌─────────────────────────────────────┐  
     │   │         Transformer Encoder         │  
     │   │ ┌─────────────────────────────────┐ │  
     │   │ │       Multi-Head Attention      │ │  
     │   │ │  ┌─────┐  ┌─────┐     ┌─────┐   │ │  
     ├──►│ │  │Head1│  │Head2│ ... │HeadN│   │ │  Specialized heads capture different     
     │   │ │  │(MP) │  │(AR) │     │(EN) │   │ │  property relationships   
     │   │ │  └─────┘  └─────┘     └─────┘   │ │  
     │   │ │             │                   │ │  
     │   │ │     Concat and Project          │ │  
     │   │ └─────────────│───────────────────┘ │  
     │   │               │                     │  
     │   │ ┌─────────────▼──────────────────┐  │  
     │   │ │      + Residual Connection     │  │  
     │   │ └─────────────│──────────────────┘  │  
     │   │               │                     │  
     │   │ ┌─────────────▼──────────────────┐  │  
     │   │ │       Layer Normalization      │  │       
     │   │ └─────────────│──────────────────┘  │  
     │   │               │                     │  
     │   │ ┌─────────────▼──────────────────┐  │  
     │   │ │     Feed-Forward Network       │  │  Element-wise transformations  
     │   │ │    [d_model → d_ff → d_model]  │  │  
     │   │ └─────────────│──────────────────┘  │  
     │   │               │                     │  
     │   │ ┌─────────────▼──────────────────┐  │  
     │   │ │      + Residual Connection     │  │  
     │   │ └─────────────│──────────────────┘  │  
     │   │               │                     │  
     │   │ ┌─────────────▼──────────────────┐  │  
     │   │ │       Layer Normalization      │  │       
     │   │ └─────────────│──────────────────┘  │  
     │   └───────────────│─────────────────────┘  
     │                   │    
     │                   ▼         
     │   ┌─────────────────────────────────────┐  
     └──►│  Repeat N layers (stack encoders)   │  
         └─────────────────────────────────────┘  
                         │    
                         ▼    
             ┌───────────────────────┐  
             │     Global Pooling    │  Combine information across    
             │                       │  all elements and properties   
             └───────────│───────────┘  
                         │    
                         ▼    
             ┌───────────────────────┐  
             │   Regression Head     │  Project to final prediction   
             │  [d_model → d_ff → 1] │  
             └───────────│───────────┘  
                         │    
                         ▼    
                  Predicted Melting Point    
                      [batch_size] 
```

This architecture implements a transformer-based model for predicting melting points from chemical composition data. The model processes input tensors representing chemical mixtures through specialized attention mechanisms and outputs melting point predictions.