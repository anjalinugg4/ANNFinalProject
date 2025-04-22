import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os



#hello 

weather_data_mapping = {
    "lightning": {
        "emotion": "intense",
        "danceability": (0.8181, 0.9090),
        "loudness": (-1.0, 0.0),
        "speechiness": (0.85, 1.00),
        "acousticness": (0.00, 0.05),
        "instrumentalness": (0.00, 0.05),
        "liveness": (0.85, 1.00),
        "valence": (0.90, 1.00),
        "tempo": (215, 245)
    },
    "rain": {
        "emotion": "melancholic",
        "danceability": (0.5454, 0.6363),
        "loudness": (-9.0, -6.0),
        "speechiness": (0.55, 0.65),
        "acousticness": (0.25, 0.35),
        "instrumentalness": (0.20, 0.30),
        "liveness": (0.55, 0.65),
        "valence": (0.08, 0.16),
        "tempo": (110, 125)
    },
    "snow": {
        "emotion": "cozy",
        "danceability": (0.0000, 0.0909),
        "loudness": (-31.0, -27.0),
        "speechiness": (0.05, 0.10),
        "acousticness": (0.85, 0.95),
        "instrumentalness": (0.80, 0.90),
        "liveness": (0.05, 0.10),
        "valence": (0.70, 0.80),
        "tempo": (55, 75)
    },
    "sandstorm": {
        "emotion": "eerie",
        "danceability": (0.7272, 0.8181),
        "loudness": (-3.0, -1.0),
        "speechiness": (0.75, 0.85),
        "acousticness": (0.05, 0.15),
        "instrumentalness": (0.05, 0.10),
        "liveness": (0.75, 0.85),
        "valence": (0.32, 0.40),
        "tempo": (195, 215)
    },
    "rime": {
        "emotion": "peaceful",
        "danceability": (0.0909, 0.1818),
        "loudness": (-36.0, -31.0),
        "speechiness": (0.00, 0.05),
        "acousticness": (0.95, 1.00),
        "instrumentalness": (0.90, 1.00),
        "liveness": (0.00, 0.05),
        "valence": (0.40, 0.50),
        "tempo": (0, 55)
    },
    "frost": {
        "emotion": "sharpness",
        "danceability": (0.3636, 0.4545),
        "loudness": (-12.0, -9.0),
        "speechiness": (0.45, 0.55),
        "acousticness": (0.35, 0.45),
        "instrumentalness": (0.30, 0.40),
        "liveness": (0.45, 0.55),
        "valence": (0.16, 0.24),
        "tempo": (140, 155)
    },
    "rainbow": {
        "emotion": "inspiring",
        "danceability": (0.9090, 1.0000),
        "loudness": (-19.0, -15.0),
        "speechiness": (0.26, 0.35),
        "acousticness": (0.55, 0.65),
        "instrumentalness": (0.50, 0.60),
        "liveness": (0.26, 0.35),
        "valence": (0.80, 0.90),
        "tempo": (155, 175)
    },
    "hail": {
        "emotion": "angry",
        "danceability": (0.6363, 0.7272),
        "loudness": (-6.0, -3.0),
        "speechiness": (0.65, 0.75),
        "acousticness": (0.15, 0.25),
        "instrumentalness": (0.10, 0.20),
        "liveness": (0.65, 0.75),
        "valence": (0.00, 0.08),
        "tempo": (175, 195)
    },
    "glaze": {
        "emotion": "elegant",
        "danceability": (0.4545, 0.5454),
        "loudness": (-23.0, -19.0),
        "speechiness": (0.18, 0.26),
        "acousticness": (0.65, 0.75),
        "instrumentalness": (0.60, 0.70),
        "liveness": (0.18, 0.26),
        "valence": (0.50, 0.60),
        "tempo": (95, 110)
    },
    "fogsmog": {
        "emotion": "edgy",
        "danceability": (0.2727, 0.3636),
        "loudness": (-15.0, -12.0),
        "speechiness": (0.35, 0.45),
        "acousticness": (0.45, 0.55),
        "instrumentalness": (0.40, 0.50),
        "liveness": (0.35, 0.45),
        "valence": (0.24, 0.32),
        "tempo": (125, 140)
    },
    "dew": {
        "emotion": "soft",
        "danceability": (0.1818, 0.2727),
        "loudness": (-27.0, -23.0),
        "speechiness": (0.10, 0.18),
        "acousticness": (0.75, 0.85),
        "instrumentalness": (0.70, 0.80),
        "liveness": (0.10, 0.18),
        "valence": (0.60, 0.70),
        "tempo": (75, 95)
    }
}


import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt

data_dir = '/Users/anjalinuggehalli/Desktop/ANN Final Project/weather'
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
test_img = 'weather/1837.jpg'
weather, emotion = predict_weather_and_emotion(test_img)
print(f"Predicted Weather: {weather} → Emotion: {emotion}")

