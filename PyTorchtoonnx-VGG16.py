import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import onnxruntime as ort

# Step 1: Load the pre-trained VGG16 model in PyTorch
model = models.vgg16(pretrained=True)
model.eval()  

# Step 2: Define image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

# Step 3: Load and preprocess an image 
image_path = "animalimg.jpg"  
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0) 

# Step 4: Perform inference with the PyTorch model
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output, 1).item()

print(f"PyTorch Model Predicted Class: {predicted_class}")

# Step 5: Convert PyTorch model to ONNX
dummy_input = torch.randn(1, 3, 224, 224) 
torch.onnx.export(model, dummy_input, "vgg16.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print("Model successfully converted to ONNX format.")

# Step 6: Evaluate ONNX model

# Load the ONNX model
onnx_model_path = "vgg16.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Convert input_tensor to NumPy format (ONNX expects NCHW format)
onnx_input = input_tensor.numpy()

# Perform inference using the ONNX model
onnx_output = ort_session.run(None, {'input': onnx_input})

# Get the predicted class from the ONNX output
onnx_predicted_class = np.argmax(onnx_output[0], axis=1)[0]

print(f"ONNX Model Predicted Class: {onnx_predicted_class}")
