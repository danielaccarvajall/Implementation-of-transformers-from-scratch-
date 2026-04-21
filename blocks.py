import torch
import torch.nn as nn
from attention import MultiHeadAttention # Importing the class you built earlier

# --- STEP 1: FeedForward (The "Thinking" Layer) ---
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

# --- STEP 2: TransformerBlock (The "Chassis") ---
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads=heads, dim_head=dim_head)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # 1. Attention Block with Residual Connection
        # Note: We use Pre-Norm (Norm -> Attn -> Add) which is standard now
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout1(attn_out)
        
        # 2. FeedForward Block with Residual Connection
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout2(ff_out)
        
        return x

# --- STEP 3: The Full Transformer (Stacking Them) ---
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim, dropout))
            
    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Test the full stack
    # dim=256, depth=6 (6 blocks deep), heads=8, mlp_dim=512
    model = Transformer(dim=256, depth=6, heads=8, dim_head=64, mlp_dim=512)
    
    dummy_input = torch.randn(1, 10, 256)
    output = model(dummy_input)
    
    print(f"Transformer Input: {dummy_input.shape}")
    print(f"Transformer Output: {output.shape}")