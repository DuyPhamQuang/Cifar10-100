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


class PatchEmbedding(nn.Module):
    '''
    Patch embedding layer that divides the input image into non-overlapping patches and projects them to a higher-dimensional space.
     - Input: (Batch_size, num_channels, height, width)
     - Output: (Batch_size, num_patches, embedding_dim)
     where num_patches = (height/patch_size) * (width/patch_size)
     and embedding_dim is the dimension of the output feature space for each patch.
    '''
    def __init__(self, input_channels, embedding_dim, patch_size):
        super().__init__()
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.patch_embedded = nn.Conv2d(input_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch_embedded(x) # (Batch_size, embedding_dim, height/patch_size, width/patch_size)
        x = x.flatten(2) # flatten the last two dimensions to get (Batch_size, embedding_dim, num_patches)
        x = x.transpose(1, 2) # transpose first and second dimensions to get (Batch_size, num_patches, embedding_dim)
        return x

class TransformerEncoderBlock(nn.Module):
    '''
    Transformer Encoder Block that consists of Multi-Head Attention and MLP layers with residual connections.
     - Input: (Batch_size, num_patches, embedding_dim)
     - Output: (Batch_size, num_patches, embedding_dim)
    '''
    def __init__(self, embedding_dim, num_heads, mlp_hidden_dim, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.multi_head_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),  
            nn.Linear(mlp_hidden_dim, embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        # Multi-Head Attention
        residual1 = x
        x = self.layer_norm1(x)  # Normalize input for attention
        x, _ = self.multi_head_attention(x, x, x)  # Self-attention
        x = self.attn_dropout(x) # Apply dropout to attention output
        x = x + residual1  # Add residual connection

        # MLP
        residual2 = x
        x = self.layer_norm2(x) # Normalize input for MLP
        x = self.mlp(x) # Pass through MLP
        x = x + residual2 # Add residual connection

        return x

class MLP_Head(nn.Module):
    '''
    MLP Head for classification that takes the output of the transformer encoder and produces class probabilities.
     - Input: (Batch_size, embedding_dim) from the CLS token
     - Output: (Batch_size, num_classes)
    '''
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(embedding_dim)
        self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.norm(x) # Normalize the CLS token output
        x = self.mlp_head(x) # Linear layer to get class logits
        return x
    

class VisionTransformer(nn.Module):
    '''
    Vision Transformer (ViT) model that combines the patch embedding, transformer encoder blocks, and MLP head for classification.
     - Input: (Batch_size, num_channels, height, width)
     - Output: (Batch_size, num_classes)
    '''

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)  
            nn.init.zeros_(module.bias)

    def __init__(self, input_channels, embedding_dim, patch_size, image_size, num_heads, mlp_hidden_dim, num_blocks, num_classes, dropout=0.1):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_embedding = PatchEmbedding(input_channels, embedding_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embedding_dim)) # Position embedding for patches and CLS token
        self.embedding_dropout = nn.Dropout(p=dropout)  # applied after adding positional embedding
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim, num_heads, mlp_hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.mlp_head = MLP_Head(embedding_dim, num_classes)
        
        # Weight initialization
        self.apply(self._init_weights)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding(x) # Get patch embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # Expand CLS token for batch size
        x = torch.cat((cls_tokens, x), dim=1) # Concatenate CLS token with patch embeddings
        x = x + self.position_embedding[:, :x.size(1), :] # Add position embedding
        x = self.embedding_dropout(x) # dropout after positional embedding

        for block in self.transformer_blocks:
            x = block(x) # Pass through transformer encoder blocks

        cls_output = x[:, 0] # Get the output corresponding to the CLS token
        output = self.mlp_head(cls_output) # Pass through MLP head for classification
        return output