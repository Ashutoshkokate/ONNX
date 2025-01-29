import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import onnxruntime as ort
import numpy as np

# Define the model using pre-trained ResNet18 with updated weights parameter
class ResNet18TransferLearning(nn.Module):
    def __init__(self):
        super(ResNet18TransferLearning, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # Output 10 classes for CIFAR-10

    def forward(self, x):
        return self.model(x)
# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Transformations for data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = ResNet18TransferLearning()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Training the PyTorch model...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluate the PyTorch model
print("Evaluating the trained PyTorch model on the test set...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the trained model on the CIFAR-10 test set: {100 * correct / total:.2f}%")

# Export model to ONNX format
onnx_file_path = "resnet18_cifar10.onnx"
dummy_input = torch.randn(1, 3, 32, 32)  # CIFAR-10 input size (1 sample, 3 channels, 32x32)
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_file_path, 
    export_params=True, 
    opset_version=11, 
    input_names=['input'], 
    output_names=['output'], 
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"Model has been saved to {onnx_file_path}")

# Load and evaluate the ONNX model
print("Evaluating the ONNX model on the test set...")
session = ort.InferenceSession(onnx_file_path)

# Get model input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Evaluate accuracy
correct = 0
total = 0

for images, labels in test_loader:
    images_np = images.numpy().astype(np.float32)
    outputs = session.run([output_name], {input_name: images_np})
    outputs_tensor = torch.tensor(outputs[0])
    _, predicted = torch.max(outputs_tensor, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Accuracy of the ONNX model on the CIFAR-10 test set: {100 * correct / total:.2f}%")
