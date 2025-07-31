import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import onnxruntime as ort
from torch.utils.tensorboard import SummaryWriter

EPOCHS = 5
BATCH_SIZE = 64
EXPORT_PATH = "mnist.onnx"

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# TensorBoard writer
writer = SummaryWriter('runs/mnist_experiment')

print("\n Training...\n")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            writer.add_scalar('Training/Loss', loss.item(), epoch * len(train_loader) + batch_idx)

    # Calculate average loss for the epoch
    avg_loss = epoch_loss / num_batches
    
    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    
    # Log to TensorBoard
    writer.add_scalar('Training/Epoch_Loss', avg_loss, epoch)
    writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

print("\n Exporting to ONNX...")
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(model, dummy_input, EXPORT_PATH, input_names=["input"], output_names=["output"])
print(f"Exported model to {EXPORT_PATH}")

# Close TensorBoard writer
writer.close()
print("TensorBoard logs saved to 'runs/mnist_experiment'")


