'''
Workflow for creating ViT model from scratch with PyTorch: 

Patch Embedding -> CLS token and Position Embedding -> Transformer Encoder -> MLP Head for Classification

1. Patch Embedding:
 - Input image is divided into non-overlapping patches (e.g., 16x16 pixels) with Conv2d layer. 
 -> (Batch_size, num_channels, height, width) -> (Batch_size, embedding_dim, height/patch_size, width/patch_size)

 - Each patch is flattened and projected to a higher-dimensional space
 -> (Batch_size, embedding_dim, height/patch_size, width/patch_size) -> (Batch_size, num_patches = (height/patch_size) * (width/patch_size), embedding_dim)

2. Transformer Encoder (N blocks):
  Residual1 = Patch (already include CLS token and Position Embedding)
  Patch -> LayerNorm1 -> MultiHeadAttention -> Add Residual1 -> X
  Residual2 = X
  X -> LayerNorm2 -> MLP -> Add Residual2 -> Output of Encoder Block
  * MLP: Linear(embedding_dim, num_hidden_nodes) -> GELU -> Linear(num_hidden_nodes, embedding_dim)

  Output of previous block is input to next block. Final output is (Batch_size, num_patches, embedding_dim)

3. MLP Head:
- CLS token (Batch_size, 1, embedding_dim) -> Normalization -> Linear Layer(embedding_dim, num_classes)
-> Output: (Batch_size, num_classes)
'''

import torch
from torch import nn
import torch.nn.functional as F

