## Model Architecture v2

```
Input Tensor [batch_size, 6, 5] ← FIXED: 6 positions (5 elements + 1 feature row)
     │    
     ▼    
┌────────────────────┐   
│  Linear Embedding  │  Transform raw features to embedding dimension      
│  [5 → d_model=128] │  ← FIXED: Reduced from 512 to 128     
└────────────────────┘        
     │    
     ▼    
Fixed Role Embedding with Permutation:
┌─────────────────────────────────────────────────────┐
│ Positions 0-4: SHUFFLED elements → Element Role     │ ← PERMUTE THESE
│                 Any element can be in any position  │
│                 Role = "I am an element"            │
│ Position 5:     Mixture properties → Feature Role   │ ← KEEP FIXED
│                 Always in position 5                │
│                 Role = "I am mixture features"      │
└─────────────────────────────────────────────────────┘
     │    
     ▼  ┌──────────────────────────────────────┐
     │  │         🚨 CRITICAL FIX 1:           │
     │  │            ┌──────────────────────────────────┐   
┌────▼──▼─────────┐  │    ACTUALLY ADD EMBEDDINGS!      │ 
│  x = x + role   │  │  x = x + role_embeddings         │
│   _embeddings   │  │  (Was commented out before)      │
└─────────────────┘  └──────────────────────────────────┘
     │    
     ▼              ┌──────────────────────────────────────┐
┌─────────────────┐ │         🚨 CRITICAL FIX 2:           │
│ Create Attention│ │      ATTENTION MASKING               │
│     Mask        │ │  mask = [True, True, False, ...]     │
│ [batch,seq_len] │ │  (True=real data, False=padding)     │
└─────────────────┘ └──────────────────────────────────────┘
     │              
     ▼   ┌─────────────────────────────────────┐  
     │   │      Transformer Encoder Block      │  ← FIXED: 2 layers (was 3)
     │   │ ┌─────────────────────────────────┐ │  
     │   │ │    Multi-Head Attention (4×)    │ │  ← FIXED: 4 heads (was 8)
     │   │ │  ┌─────┐  ┌─────┐  ┌─────┐      │ │  
     ├──►│ │  │Head1│  │Head2│  │Head3│ ...  │ │  Specialized heads for    
     │   │ │  │(MP) │  │(AR) │  │(EN) │      │ │  different properties   
     │   │ │  └─────┘  └─────┘  └─────┘      │ │  
     │   │ │             │                   │ │  ┌─────────────────────┐
     │   │ │    🔍 WITH ATTENTION MASK       │ │  │  mask prevents model │
     │   │ │    (ignores padding tokens)     │ │  │  from learning from  │
     │   │ │             │                   │ │  │  padding positions   │
     │   │ │     Concat and Project          │ │  └─────────────────────┘
     │   │ └─────────────│───────────────────┘ │  
     │   │               │                     │  
     │   │ ┌─────────────▼─────────────────┐   │  
     │   │ │   + Residual Connection       │   │  
     │   │ └─────────────│─────────────────┘   │  
     │   │               │                     │  
     │   │ ┌─────────────▼─────────────────┐   │  
     │   │ │    Layer Normalization        │   │       
     │   │ └─────────────│─────────────────┘   │  
     │   │               │                     │  
     │   │ ┌─────────────▼─────────────────┐   │  
     │   │ │  Feed-Forward Network         │   │  ← Element-wise transformations  
     │   │ │  [128 → 512 → 128]            │   │     with dropout=0.4
     │   │ │  + Dropout(0.4)               │   │  ← FIXED: Increased dropout
     │   │ └─────────────│─────────────────┘   │  
     │   │               │                     │  
     │   │ ┌─────────────▼─────────────────┐   │  
     │   │ │   + Residual Connection       │   │  
     │   │ └─────────────│─────────────────┘   │  
     │   │               │                     │  
     │   │ ┌─────────────▼─────────────────┐   │  
     │   │ │    Layer Normalization        │   │       
     │   │ └─────────────│─────────────────┘   │  
     │   └───────────────│─────────────────────┘  
     │                   │    
     │                   ▼         
     │   ┌─────────────────────────────────────┐  
     └──►│  Stack 2 Layers (was 3)             │  ← FIXED: Reduced complexity
         │  🔧 WITH WEIGHT DECAY = 0.01        │  ← FIXED: Added L2 regularization
         └─────────────────────────────────────┘  
                         │    
                         ▼    
             ┌───────────────────────┐  
             │   🎯 Global Pooling   │  Combine information across    
             │   (Mean over valid    │  ONLY non-masked positions
             │    positions only)    │  (ignores padding)
             └───────────│───────────┘  
                         │    
                         ▼    
             ┌───────────────────────┐  
             │   Regression Head     │  Project to final prediction   
             │  [128 → 128 → 1]      │  ← 2 layers with dropout
             │  + Dropout(0.4)       │  
             └───────────│───────────┘  
                         │    
                         ▼    
                  Predicted Melting Point    
                     [batch_size] 
```

This architecture implements a transformer-based model for predicting melting points from chemical composition data. The model processes input tensors representing chemical mixtures through specialized attention mechanisms and outputs melting point predictions.



# Element Position Permutation: Why It Matters

Understanding how element positioning affects model training is crucial for robust alloy property prediction.

## Without Permutation (Current - BROKEN)

### The Problem
When elements are always in the same positions during training, the model incorrectly learns **positional dependencies** instead of **elemental properties**.

```
Training Examples:
   AlFeNi → Position 0=Al, Position 1=Fe, Position 2=Ni
   CuZnSn → Position 0=Cu, Position 1=Zn, Position 2=Sn

Model Learns:
   "Position 0 behaves like Al/Cu"
   "Position 1 behaves like Fe/Zn" 
   "Position 2 behaves like Ni/Sn"
```

### When Testing Fails

```
Pentanary Test: AlFeCuNiCr

Model's Broken Logic:
   Position 0 = Al (correct by chance)
   Position 1 = Fe (correct by chance)  
   Position 2 = Cu (WRONG! expects Ni-like behavior)
   Position 3 = ? (complete panic - never seen 4+ elements!)
   Position 4 = ? (total confusion)
```

---

## With Permutation (Fixed - CORRECT)

### The Solution
By randomly shuffling element positions during training, we force the model to learn **true elemental properties** regardless of position.

```
Training Examples (Multiple Permutations):
   AlFeNi → Position 0=Fe, Position 1=Al, Position 2=Ni  (shuffled)
   CuZnSn → Position 1=Sn, Position 2=Cu, Position 0=Zn  (shuffled)
   AlFeNi → Position 2=Al, Position 0=Ni, Position 1=Fe  (shuffled again)
   CuZnSn → Position 0=Cu, Position 2=Zn, Position 1=Sn  (different shuffle)

Model Learns:
   "Al has these properties regardless of position"
   "Fe has these properties regardless of position"
   "Any position can contain any element"
```

### Robust Testing Results

```
Pentanary Test: AlFeCuNiCr (any order)

Model's Smart Logic:
   "I detect Al properties → apply Al behavior"
   "I detect Fe properties → apply Fe behavior"  
   "I detect Cu properties → apply Cu behavior"
   "I detect Ni properties → apply Ni behavior"
   "I detect Cr properties → apply Cr behavior"
   
Position-independent understanding!
```

---

## Key Benefits

| Aspect | Without Permutation | With Permutation |
|--------|-------------------|------------------|
| **Scalability** | Breaks with more elements | Handles any number of elements |
| **Generalization** | Position-dependent | Position-independent |
| **Real-world applicability** | Limited to training patterns | Works with any composition |
| **Model robustness** | Brittle and unreliable | Stable and predictable |

> **Note**: Permutation training is essential for any model that needs to understand **compositional properties** rather than **sequential patterns**.