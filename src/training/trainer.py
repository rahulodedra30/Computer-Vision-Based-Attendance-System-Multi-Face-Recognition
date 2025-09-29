"""
Training Script
Run from project root: python src/training/trainer.py
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

# Add src to path (we're in src/training/, need to go up one level to src/)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.simple_cnn import create_model, count_parameters

class SimpleDataset(Dataset):
    """Simple dataset class."""
    
    def __init__(self, img_dir, identity_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        print(f"Loading identity file: {identity_file}")
        # Read identity file
        self.data = pd.read_csv(identity_file, sep=' ', header=None, names=['filename', 'identity'])
        print(f"Identity file loaded with {len(self.data)} entries")
        
        # Only keep files that actually exist
        print("Checking which images exist...")
        existing_files = []
        for idx, row in self.data.iterrows():
            img_path = os.path.join(img_dir, row['filename'])
            if os.path.exists(img_path):
                existing_files.append(row)
        
        self.data = pd.DataFrame(existing_files)
        print(f"Found {len(self.data)} existing images")
        
        if len(self.data) == 0:
            raise ValueError("No matching images found!")
        
        # Create label mapping
        unique_ids = sorted(self.data['identity'].unique())
        self.id_to_label = {id_val: i for i, id_val in enumerate(unique_ids)}
        self.num_classes = len(unique_ids)
        
        print(f"Number of unique celebrities: {self.num_classes}")
    
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
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.id_to_label[row['identity']]
        
        return image, label

def main():
    print("=== Simple CNN Training ===")
    
    img_dir = "scripts/dataset/celeba-25000/img_align_celeba"
    identity_file = "scripts/dataset/metadata/identity_CelebA.txt"

    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for images in: {os.path.abspath(img_dir)}")
    print(f"Looking for identity file: {os.path.abspath(identity_file)}")
    
    # Check paths exist
    if not os.path.exists(img_dir):
        print(f"Images directory not found!")
        print("Available directories in scripts/dataset/:")
        if os.path.exists("scripts/dataset/"):
            for item in os.listdir("scripts/dataset/"):
                item_path = os.path.join("scripts/dataset/", item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}")
        return
    
    if not os.path.exists(identity_file):
        print(f"Identity file not found!")
        print("Checking alternative locations...")
        alt_locations = [
            "scripts/dataset/metadata/identity_CelebA.txt",
            "scripts/dataset/celeba-dataset.zip",  # If not extracted
        ]
        for alt in alt_locations:
            if os.path.exists(alt):
                print(f"  ✓ Found: {alt}")
            else:
                print(f" Not found: {alt}")
        return
    
    print("✅ All paths found!")
    
    # Device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    # To this (force CPU):
    device = torch.device('cpu')
    print(f"Using device: {device} (forced CPU due to CUDA compatibility)")
    
    
    # Transforms
    print("\nSetting up transforms...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    print("\nLoading dataset...")
    try:
        dataset = SimpleDataset(img_dir, identity_file, transform)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    if len(dataset) < 10:
        print(f"⚠️  Only {len(dataset)} samples found. Need more data for training.")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples: {len(val_dataset)}")
    
    # Data loaders
    print("\nCreating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Model
    print(f"\nCreating model with {dataset.num_classes} classes...")
    model = create_model(dataset.num_classes).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nStarting training...")
    num_epochs = 5
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
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
            
            if batch_idx % 5 == 0:
                print(f'  Batch {batch_idx}: Loss {loss.item():.4f}')
        
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
        
        # Calculate accuracies
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Results:")
        print(f"  Train: Loss {train_loss/len(train_loader):.4f}, Acc {train_acc:.2f}%")
        print(f"  Val:   Loss {val_loss/len(val_loader):.4f}, Acc {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  🎉 New best validation accuracy: {val_acc:.2f}%")
            
            # Save model
            os.makedirs('results', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': dataset.num_classes,
                'accuracy': val_acc
            }, 'results/best_model.pth')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("Model saved to: results/best_model.pth")
    print("="*50)

if __name__ == "__main__":
    main()