import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import EmotionCNN
import os
import numpy as np

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Using heavy data augmentation to prevent overfitting
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder('train', transform=train_transform)
    test_dataset = datasets.ImageFolder('test', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Calculate class weights for Imbalanced learning
    class_counts = [len(os.listdir(os.path.join('train', c))) for c in train_dataset.classes]
    total_samples = sum(class_counts)
    class_weights = [total_samples / c for c in class_counts]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = EmotionCNN(num_classes=len(train_dataset.classes)).to(device)

    # Using cross entropy loss with weights
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Since ResNet18 is quite large, lowering the learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    # Using a learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    epochs = 20

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss/len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss/len(test_loader)
        val_acc = 100. * correct / total

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'emotion_model.pth')
            print("  -> Saved best model")

    print("Model saved to emotion_model.pth")

    # Save classes mapping
    with open('classes.txt', 'w') as f:
        f.write('\n'.join(train_dataset.classes))

if __name__ == "__main__":
    train()
