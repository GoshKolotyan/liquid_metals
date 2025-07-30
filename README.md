# Liquid Metals

ML-based calculation for predicting Melting Point for Liquid Metals 🔥

## Project Overview

This project uses machine learning techniques to predict the melting points of liquid metals based on their properties. The implementation incorporates transformer architecture for advanced feature extraction and prediction accuracy.

## TODOS

### Finish tokenizer
- [x] Finish tokenizer logic (add feature computing class)
- [x] Add support for various metal representations
- [x] Implement normalization for input features

### Augmentation
- [x] Implement logic for augmentation (permutation logic, adding noise)
- [x] Create dataset expansion utilities
- [x] Add validation for augmented data

### Implement Transformer
- [x] Implement Attention
- [x] Test Attention
- [x] Implement MultiHeadAttention
- [x] Build model (Attention, RegressionHead)
- [x] Create Dataloader and Dataset classes
- [x] Test before Training (Is attention working as expected)