# Liquid Metals

ML-based calculation for predicting Melting Point for Liquid Metals 🔥

## Project Overview

This project uses machine learning techniques to predict the melting points of liquid metals based on their properties. The implementation incorporates transformer architecture for advanced feature extraction and prediction accuracy.

## TODOS

### Finish tokenizer
- [ ] Finish tokenizer logic (add feature computing class)
- [ ] Add support for various metal representations
- [ ] Implement normalization for input features

### Augmentation
- [ ] Implement logic for augmentation (permutation logic, adding noise)
- [ ] Create dataset expansion utilities
- [ ] Add validation for augmented data

### Implement Transformer
- [ ] Implement Attention
- [ ] Test Attention
- [ ] Implement MultiHeadAttention
- [ ] Build model (Attention, RegressionHead)
- [x] Create Dataloader and Dataset classes
- [ ] Test before Training (Is attention working as expected)