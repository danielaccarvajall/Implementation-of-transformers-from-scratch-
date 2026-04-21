import torch
import torch.nn as nn
from einops import rearrange
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5 # 1/sqrt(d_k) from the formula
        
        inner_dim = heads * dim_head
        
        # One big matrix multiplication for Q, K, V
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim) -> e.g., (32, 100, 512)
        
        # 1. Get Q, K, V
        # Chunk splits the big tensor into 3 parts (Q, K, V)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 2. Scaled Dot-Product Attention
        # math: Q * K^T / sqrt(d)
        # einops allows 'einsum' style dot products, usually easier to just use matmul here
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 3. Softmax
        attn = dots.softmax(dim=-1)

        # 4. Multiply by V
        out = torch.matmul(attn, v)
        
        # 5. Merge heads back together (The Reverse of Step 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Create a random tensor: Batch=1, Sequence Length=10, Dimension=256
    dummy_input = torch.randn(1, 10, 256)
    
    # Initialize Attention
    attention_layer = MultiHeadAttention(dim=256, heads=8, dim_head=64)
    
    # Forward pass
    output = attention_layer(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}") 
    # If this prints (1, 10, 256), YOU SUCCEEDED.