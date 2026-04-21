import torch
import torchvision.transforms as transforms
from PIL import Image
from vit import ViT

# 1. Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Load the Architecture (Must match training exactly)
model = ViT(
    image_size=32,
    patch_size=4,
    num_classes=10,
    dim=128,
    depth=6,
    heads=8,
    mlp_dim=256,
    dropout=0.1
).to(device)

# 3. Load the Weights
print("Loading model weights...")
try:
    model.load_state_dict(torch.load('vit_cifar10.pth', map_location=device))
    model.eval() # CRITICAL: Switch to evaluation mode (turns off Dropout)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: 'vit_cifar10.pth' not found. Did you finish training?")
    exit()

# 4. Prepare the Image
# CIFAR-10 images are tiny (32x32), so we must resize our input to match
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict(image_path):
    try:
        img = Image.open(image_path)
        img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension -> (1, 3, 32, 32)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            # The output is raw numbers (logits). We take the max to find the prediction.
            _, predicted = torch.max(outputs, 1)
            
        print(f"Prediction for {image_path}: {classes[predicted.item()]}")
        
    except Exception as e:
        print(f"Could not process image: {e}")

# --- RUN IT ---
if __name__ == "__main__":
    # Replace this with the name of an image you have
    predict("test_image.jpg")