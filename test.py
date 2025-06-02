import cv2
import torch
from torchvision import transforms
from model import CatDogCNN
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Predict a Single Image ---
img_path = r'C:\Users\KIIT\DL-Projects\cat_dog_classifier3\test_images\dogs\9931.jpg'
img_path1 = r'C:\Users\KIIT\DL-Projects\cat_dog_classifier3\test_images\cats\cat112.jpg'

img = cv2.imread(img_path1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img_tensor = transform(img).unsqueeze(0).to(device)

model = CatDogCNN().to(device)
model.load_state_dict(torch.load('resnet_catdog.pth'))
model.eval()

with torch.no_grad():
    output = model(img_tensor)
    prediction = 'Dog' if output.item() > 0.5 else 'Cat'
print(f"Predicted: {prediction}")

# --- Accuracy on Full Test Dataset ---
# Folder structure: test_images/cats/*.jpg, test_images/dogs/*.jpg
test_data_path = r'C:\Users\KIIT\DL-Projects\cat_dog_classifier3\test_images'
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
test_dataset = ImageFolder(test_data_path, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
