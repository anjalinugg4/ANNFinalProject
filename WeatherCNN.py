import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt

data_dir = '/Users/anjalinuggehalli/Desktop/ANNFinalProject/weather'
batch_size = 32
num_epochs = 1
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

# 5. Simple CNN Model
class WeatherCNN(nn.Module):
    def __init__(self, num_classes):
        super(WeatherCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 64, 64]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 32, 32]
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
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

# 8. Predict & Map to Emotion
def predict_weather_and_emotion(image_path):
    from PIL import Image
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    pred_idx = output.argmax(dim=1).item()
    weather_label = dataset.classes[pred_idx]
    emotion = weather_to_emotion.get(weather_label, "unknown")
    return weather_label, emotion

# Example usage
test_img = 'weather/lightning/1837.jpg'
weather, emotion = predict_weather_and_emotion(test_img)
print(f"Predicted Weather: {weather} → Emotion: {emotion}")

