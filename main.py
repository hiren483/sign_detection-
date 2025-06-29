import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and preprocess the data
train_df = pd.read_csv("sign_mnist_train.csv")
test_df = pd.read_csv("sign_mnist_test.csv")

train_labels = train_df["label"]
train_features = train_df.drop("label" , axis=1)

train_labels = torch.tensor(train_labels.values, dtype=torch.long)
train_features = torch.tensor(train_features.values, dtype=torch.float32) / 255.0 

# Define dataset class
class signdataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        images = self.features[i].reshape(1,28,28)
        labels = self.labels[i]
        return images , labels

# Dataloaders
train_loader = torch.utils.data.DataLoader(signdataset(train_features, train_labels), batch_size=64, shuffle=True)


# Define the model
class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*7*7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

#hyper parameters 
num_classes = 26
n_total_steps = len(train_loader)

# Train and evaluate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/5], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

# Testing
model.eval()

#test data
test_labels = test_df["label"]
test_features = test_df.drop("label" , axis=1)

test_labels = torch.tensor(test_labels.values, dtype=torch.long)
test_features = torch.tensor(test_features.values, dtype=torch.float32) / 255.0

# Dataloaders
test_loader = torch.utils.data.DataLoader(signdataset(test_features, test_labels), batch_size=64, shuffle=True)


correct = 0
total = 0
sample_logged = False
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if not sample_logged:
            print("Raw output logits for one sample:", outputs[0])
            print("Predicted class:", predicted[0].item())
            print("Actual label:", labels[0].item())
            sample_logged = True

accuracy = 100 * correct / total
print(f"accuracy : {accuracy}")

torch.save(model.state_dict(), "sign_model.pth")