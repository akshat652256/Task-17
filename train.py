import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn  # Corrected import for nn module
from model import LeNet5
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader

# Transformations to normalize the dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
trainset, valset = random_split(dataset, [train_size, val_size])
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# Set device and load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
best_val_f1 = 0.0
patience = 3
counter = 0

# Lists to store performance metrics
train_losses, val_losses = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []

# Training loop
for epoch in range(5):  # Train for 5 epochs
    model.train()
    all_preds, all_labels = [], []
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate training metrics
    train_loss = running_loss / len(trainloader)
    train_f1 = f1_score(all_labels, all_preds, average='macro')
    train_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    train_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Store training metrics
    train_losses.append(train_loss)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)

    # Validation loop
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    # Calculate validation metrics
    val_loss /= len(valloader)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
    val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
    
    # Store validation metrics
    val_losses.append(val_loss)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    
    # Print the results for the current epoch
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")
    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
    
    # Early stopping based on validation F1 score
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        counter = 0
        torch.save(model.state_dict(), "lenet5.pth")  # Save best model
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

print("Training Complete! Best model saved as lenet5.pth")
