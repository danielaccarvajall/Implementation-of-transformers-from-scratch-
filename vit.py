import torch
import torch.nn as nn
from einops import rearrange, repeat
from blocks import Transformer # Import the stack you just built

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0.):
        super().__init__()
        
        # 1. Calculate number of patches
        # Example: 256x256 image with 32x32 patches = (8*8) = 64 patches
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        # 2. Patch Embedding Layer
        # This acts as the "Tokenizer" for images
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Sequential(
            # Project flattened patch to 'dim' (e.g. 768*1 -> 512)
            nn.Linear(patch_dim, dim),
        )

        # 3. Positional Embeddings & CLS Token
        # We need to add "position" info so the model knows top-left vs bottom-right
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # Special token for classification
        self.dropout = nn.Dropout(dropout)

        # 4. The Transformer Encoder (The Brain)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 5. Classification Head (The Output)
        # Takes the "CLS" token and predicts the class (Dog, Cat, etc.)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # img shape: (batch, channels, height, width)
        
        # 1. Chunk Image into Patches using Einops
        # 'p' is patch size. We flatten (c h p) into one vector.
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        
        # 2. Project to Embedding Dimension
        x = self.to_patch_embedding(x) # shape: (b, num_patches, dim)
        
        # 3. Add CLS Token
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1) # shape: (b, num_patches + 1, dim)
        
        # 4. Add Position Embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # 5. Run Transformer
        x = self.transformer(x)

        # 6. Classification
        # We only care about the first token (CLS token) for the prediction
        x = x[:, 0]
        
        return self.mlp_head(x)

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Create a random "fake image" (Batch=1, Channels=3, Height=256, Width=256)
    fake_image = torch.randn(1, 3, 256, 256)
    
    # Initialize ViT
    model = ViT(
        image_size=256,
        patch_size=32,    # Break image into 32x32 squares
        num_classes=1000, # ImageNet classes
        dim=512,          # Vector size
        depth=6,          # 6 Transformer blocks
        heads=8,
        mlp_dim=1024
    )
    
    output = model(fake_image)
    
    print(f"Image Input: {fake_image.shape}")
    print(f"ViT Output (Logits): {output.shape}") # Should be (1, 1000)