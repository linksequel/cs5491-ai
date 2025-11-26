"""
Music Genre Classification using CNN (PyTorch)
===============================================
This script uses Convolutional Neural Networks to classify music genres
based on Mel Spectrogram images.

Dataset: GTZAN Mel Spectrogram images (10 genres × 100 images each)
Split: Files 0-79 for training, Files 80-99 for testing
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# ========== Configuration ==========
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, '../input/gtzan-dataset-music-genre-classification/Data/images_original')
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')

# Genre list (alphabetically sorted to match folder order)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)  # ResNet expects 224x224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_TRANSFER_LEARNING = True  # Set to True to use pre-trained ResNet18

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)


class MelSpectrogramDataset(Dataset):
    """Custom Dataset for Mel Spectrogram images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_data():
    """
    Load image paths and labels.
    Split: files 0-79 for train, 80-99 for test (same as feature-based approach)
    """
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    
    for genre_idx, genre in enumerate(GENRES):
        genre_path = os.path.join(DATA_PATH, genre)
        
        for i in range(100):
            # Image filename format: blues00045.png
            img_name = f"{genre}{i:05d}.png"
            img_path = os.path.join(genre_path, img_name)
            
            if os.path.exists(img_path):
                if i < 80:
                    train_paths.append(img_path)
                    train_labels.append(genre_idx)
                else:
                    test_paths.append(img_path)
                    test_labels.append(genre_idx)
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Testing samples: {len(test_paths)}")
    
    return train_paths, train_labels, test_paths, test_labels


class MusicGenreCNN(nn.Module):
    """
    CNN model for music genre classification.
    Architecture: Conv2D -> MaxPool -> Conv2D -> MaxPool -> Conv2D -> MaxPool -> FC -> FC -> Output
    """
    
    def __init__(self, num_classes=10):
        super(MusicGenreCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 -> 64
            nn.Dropout2d(0.25),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            nn.Dropout2d(0.25),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            nn.Dropout2d(0.25),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 -> 8
            nn.Dropout2d(0.25),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class MusicGenreResNet(nn.Module):
    """
    Transfer Learning model using pre-trained ResNet18.
    Much better for small datasets!
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        super(MusicGenreResNet, self).__init__()
        
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Freeze early layers (optional, can unfreeze for fine-tuning)
        for param in list(self.resnet.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(test_losses, label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(train_accs, label='Train Accuracy')
    axes[1].plot(test_accs, label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'cnn_training_history.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_PATH, 'cnn_training_history.png')}")


def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('CNN Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'cnn_confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_PATH, 'cnn_confusion_matrix.png')}")


def main():
    print("="*60)
    print("MUSIC GENRE CLASSIFICATION - CNN (PyTorch)")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    train_paths, train_labels, test_paths, test_labels = load_data()
    
    # Step 2: Create data transforms and datasets
    print("\n[2/5] Preparing datasets...")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for testing
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MelSpectrogramDataset(train_paths, train_labels, train_transform)
    test_dataset = MelSpectrogramDataset(test_paths, test_labels, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    
    # Step 3: Create model, loss function, optimizer
    print("\n[3/5] Creating model...")
    if USE_TRANSFER_LEARNING:
        print("Using Transfer Learning (ResNet18 pre-trained on ImageNet)")
        model = MusicGenreResNet(num_classes=len(GENRES), pretrained=True).to(DEVICE)
    else:
        print("Using Custom CNN (training from scratch)")
        model = MusicGenreCNN(num_classes=len(GENRES)).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Step 4: Training loop
    print("\n[4/5] Training model...")
    print("-" * 60)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Evaluate
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, DEVICE)
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, 'best_cnn_model.pth'))
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # Step 5: Final evaluation
    print("\n[5/5] Final evaluation...")
    print("-" * 60)
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_PATH, 'best_cnn_model.pth')))
    
    # Final evaluation
    _, final_acc, y_pred, y_true = evaluate(model, test_loader, criterion, DEVICE)
    
    print(f"\n{'='*60}")
    print(f"BEST TEST ACCURACY: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"{'='*60}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=GENRES))
    
    # Save plots
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    plot_confusion_matrix(y_true, y_pred, GENRES)
    
    # Save results
    results = {
        'best_accuracy': best_acc,
        'final_accuracy': final_acc,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    }
    
    print("\n" + "="*60)
    print("CNN TRAINING COMPLETE")
    print(f"Best Test Accuracy: {best_acc:.2%}")
    print(f"Model saved to: {os.path.join(OUTPUT_PATH, 'best_cnn_model.pth')}")
    print("="*60)
    
    return model, results


if __name__ == "__main__":
    model, results = main()

