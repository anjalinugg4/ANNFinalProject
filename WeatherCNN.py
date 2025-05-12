import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image



data_dir = '/Users/anjalinuggehalli/Desktop/ANNFinalProject/weather'
batch_size = 32
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Mapping from weather label to emotion
weather_to_emotion = {
    "lightning": "intense",
    "rain": "melancholic",
    "snow": "cozy",
    "sandstorm": "eerie",
    "rime": "peaceful",
    "frost": "sharpness",
    "rainbow": "inspiring",
    "hail": "angry",
    "glaze": "elegant",
    "fogsmog": "edgy",
    "dew": "soft"
}

# 3. Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 4. Dataset and Dataloader
dataset = ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class WeatherCNN(nn.Module):
    def __init__(self, num_classes):
        super(WeatherCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Input size changes due to 3 poolings
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> [B, 32, 64, 64]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> [B, 64, 32, 32]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> [B, 128, 16, 16]
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# 6. Training Setup
num_classes = len(dataset.classes)
model = WeatherCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")


def predict_weather_and_emotion(image_path):
    from PIL import Image
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)
        confidence = confidence.item()
    
    weather_label = dataset.classes[pred_idx.item()]
    emotion = weather_to_emotion.get(weather_label, "unknown")
    
    return weather_label, emotion, confidence



def plot_confidences(image_path, model, dataset, transform, threshold=0.9):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()

    classes = dataset.classes
    plt.figure(figsize=(10, 5))
    bars = plt.bar(classes, probs, color=['green' if p >= threshold else 'skyblue' for p in probs])
    plt.axhline(y=threshold, color='red', linestyle='--', label='90% Threshold')
    plt.ylabel('Confidence (Probability)')
    plt.xlabel('Weather Class')
    plt.title('Model Prediction Confidence per Class')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()




