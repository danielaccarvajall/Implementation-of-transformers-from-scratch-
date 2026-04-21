import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from vit import ViT

# Keep imports and class definitions at the top level
# But EXECUTION code must be inside the main guard

if __name__ == "__main__":
    # 1. Setup Device (Use GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Prepare Data (CIFAR-10: 32x32 color images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Downloading dataset...")
    # It is safe to create the dataset object here
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # CRITICAL: This DataLoader uses num_workers=2, which caused the crash
    # It works now because it's inside the __name__ == "__main__" block
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # 3. Initialize Model
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

    # 4. Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # 5. Training Loop
    print("Starting training...")
    for epoch in range(5):  
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), 'vit_cifar10.pth')
    print("Model saved to vit_cifar10.pth")