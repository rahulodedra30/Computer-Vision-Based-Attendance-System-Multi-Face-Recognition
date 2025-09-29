"""
Week 1: Training script using real CelebA celebrity IDs.
No CSV processing needed - uses existing identity file directly.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.simple_cnn import create_model, count_parameters

class CelebrityDataset(Dataset):
    """Dataset for real celebrity identification."""
    
    def __init__(self, img_dir, identity_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        print(f"Loading celebrity identities: {identity_file}")
        # Read identity file (image_id, celebrity_id)
        self.data = pd.read_csv(identity_file, sep=' ', header=None, 
                               names=['filename', 'celebrity_id'])
        print(f"Loaded {len(self.data)} image-celebrity mappings")
        
        # Filter for existing images only
        print("Checking which images exist...")
        existing_data = []
        for _, row in self.data.iterrows():
            img_path = os.path.join(img_dir, row['filename'])
            if os.path.exists(img_path):
                existing_data.append(row)
        
        self.data = pd.DataFrame(existing_data)
        print(f"Found {len(self.data)} existing images")
        
        if len(self.data) == 0:
            raise ValueError("No images found!")
        
        # Get unique celebrities and create mapping to sequential IDs
        unique_celebs = sorted(self.data['celebrity_id'].unique())
        self.celeb_to_idx = {celeb_id: idx for idx, celeb_id in enumerate(unique_celebs)}
        self.num_celebrities = len(unique_celebs)
        
        # Map celebrity IDs to sequential indices (0, 1, 2, ...)
        self.data['label'] = self.data['celebrity_id'].map(self.celeb_to_idx)
        
        # Statistics
        celeb_counts = self.data['celebrity_id'].value_counts()
        print(f"Dataset statistics:")
        print(f"  Total celebrities: {self.num_celebrities}")
        print(f"  Celebrity ID range: {min(unique_celebs)} to {max(unique_celebs)}")
        print(f"  Images per celebrity - Min: {celeb_counts.min()}, Max: {celeb_counts.max()}, Mean: {celeb_counts.mean():.1f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        label = int(row['label'])
        return image, label


def main():
    print("=== Week 1: Celebrity Identification CNN Training ===")
    
    # Configuration - Update to match your dataset location
    dataset_dir = "scripts/dataset/celeba-100"  # Change based on your num_images
    img_dir = os.path.join(dataset_dir, "img_align_celeba")
    train_file = os.path.join(dataset_dir, "train_identity.txt")
    val_file = os.path.join(dataset_dir, "val_identity.txt")
    
    # Training parameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    
    print(f"Dataset: {dataset_dir}")
    print(f"Images: {img_dir}")
    
    # Check if dataset exists
    if not os.path.exists(img_dir):
        print("Dataset not found!")
        print("\nTo create dataset, run:")
        print("cd scripts && python download_data.py --data-dir ./dataset --num-images 10000")
        return
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print("Train/val split files not found!")
        return
    
    # Device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Transforms
    print("\nSetting up transforms...")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("\nLoading datasets...")
    try:
        train_dataset = CelebrityDataset(img_dir, train_file, train_transform)
        val_dataset = CelebrityDataset(img_dir, val_file, val_transform)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Number of celebrities: {train_dataset.num_celebrities}")
        
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        return
    
    # Data loaders
    print("\nCreating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    print(f"\nCreating CNN for {train_dataset.num_celebrities}-class classification...")
    model = create_model(train_dataset.num_celebrities).to(device)
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60)
    
    best_val_acc = 0
    os.makedirs('results', exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 20 == 0:
                current_acc = 100. * train_correct / train_total
                print(f'  Batch {batch_idx}: Loss {loss.item():.4f}, Acc {current_acc:.1f}%')
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        
        # Store history
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f"Results:")
        print(f"  Train: Loss {epoch_train_loss:.4f}, Acc {epoch_train_acc:.2f}%")
        print(f"  Val:   Loss {epoch_val_loss:.4f}, Acc {epoch_val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(epoch_val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            print(f"  ★ New best: {epoch_val_acc:.2f}%")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'num_celebrities': train_dataset.num_celebrities,
                'accuracy': epoch_val_acc,
                'epoch': epoch,
                'celebrity_mapping': train_dataset.celeb_to_idx
            }, 'results/best_celebrity_model.pth')
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'num_epochs': num_epochs,
        'num_celebrities': train_dataset.num_celebrities,
        'total_params': total_params,
        'dataset_size': len(train_dataset) + len(val_dataset),
        'training_date': datetime.now().isoformat()
    }
    
    with open('results/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\n" + "="*60)
    print("WEEK 1 TRAINING COMPLETED!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Celebrities identified: {train_dataset.num_celebrities}")
    print(f"Model saved: results/best_celebrity_model.pth")
    print(f"History saved: results/training_history.json")
    print("="*60)


if __name__ == "__main__":
    main()