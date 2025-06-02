import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from model import CatDogCNN  # or your ResNet-based model

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to test image
img_path = r'C:\Users\KIIT\DL-Projects\cat_dog_classifier3\test_images\cats\cat112.jpg'

# Load and preprocess image
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img_tensor = transform(img_rgb).unsqueeze(0).to(device)

# Load model
model = CatDogCNN().to(device)
model.load_state_dict(torch.load('resnet_catdog.pth'))
model.eval()

# Predict
with torch.no_grad():
    output = model(img_tensor)
    prediction = 'Dog' if output.item() > 0.5 else 'Cat'

# 🔍 Show image and prediction
plt.imshow(img_rgb)
plt.title(f"Predicted: {prediction}", fontsize=16)
plt.axis('off')
plt.show()
