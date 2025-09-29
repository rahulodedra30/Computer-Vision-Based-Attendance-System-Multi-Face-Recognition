"""
Week 1: CNN model for real celebrity identification.
Uses actual CelebA celebrity IDs for individual recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CelebrityIdentificationCNN(nn.Module):
    """CNN for multi-class celebrity identification using real CelebA data."""
    
    def __init__(self, num_celebrities):
        super(CelebrityIdentificationCNN, self).__init__()
        
        self.num_celebrities = num_celebrities
        
        # Smaller convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        
        # Smaller fully connected layers
        # After 3 pooling: 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_celebrities)

    # def __init__(self, num_celebrities):
    #     super(CelebrityIdentificationCNN, self).__init__()
        
    #     self.num_celebrities = num_celebrities
        
    #     # Convolutional layers
    #     self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    #     self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    #     self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    #     self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
    #     # Batch normalization
    #     self.bn1 = nn.BatchNorm2d(64)
    #     self.bn2 = nn.BatchNorm2d(128)
    #     self.bn3 = nn.BatchNorm2d(256)
    #     self.bn4 = nn.BatchNorm2d(512)
        
    #     # Pooling and dropout
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.dropout = nn.Dropout(0.5)
        
    #     # Fully connected layers
    #     self.fc1 = nn.Linear(512 * 14 * 14, 1024)
    #     self.fc2 = nn.Linear(1024, 512)
    #     self.classifier = nn.Linear(512, num_celebrities)
    
    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 224->112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 112->56  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 56->28
        # x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 28->14
        
        # Flatten
        x = x.view(-1, 512 * 14 * 14)
        
        # Classification layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


def create_model(num_celebrities):
    """Create CNN model for celebrity identification."""
    return CelebrityIdentificationCNN(num_celebrities)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def extract_celebrity_images(model, target_image_tensor, all_images_tensors, 
                           image_filenames, top_k=20):
    """
    Extract similar celebrity images using the trained model features.
    This supports the Week 1 requirement for celebrity selection.
    """
    model.eval()
    
    with torch.no_grad():
        # Get features for target image
        target_features = model.extract_features(target_image_tensor.unsqueeze(0))
        
        # Get features for all images in dataset
        similarities = []
        
        for i, img_tensor in enumerate(all_images_tensors):
            img_features = model.extract_features(img_tensor.unsqueeze(0))
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(target_features, img_features, dim=1)
            similarities.append((image_filenames[i], similarity.item()))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def model_summary(model, input_size=(3, 224, 224)):
    """Print model architecture summary."""
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {
                "input_shape": list(input[0].size()),
                "output_shape": list(output.size()),
                "trainable_params": sum([p.numel() for p in module.parameters() if p.requires_grad]),
                "params": sum([p.numel() for p in module.parameters()])
            }
        
        if not isinstance(module, (nn.Sequential, nn.ModuleList)) and not (module == model):
            hooks.append(module.register_forward_hook(hook))
    
    # Create summary dict
    summary = {}
    hooks = []
    
    # Register hooks
    model.apply(register_hook)
    
    # Run forward pass
    device = next(model.parameters()).device
    x = torch.randn(1, *input_size).to(device)
    model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Print summary
    print("="*70)
    print(f"{'Layer (type)':<20} {'Output Shape':<20} {'Param #':<15}")
    print("="*70)
    
    total_params = 0
    total_trainable_params = 0
    
    for layer_name, layer_info in summary.items():
        print(f"{layer_name:<20} {str(layer_info['output_shape']):<20} {layer_info['params']:<15,}")
        total_params += layer_info['params']
        total_trainable_params += layer_info['trainable_params']
    
    print("="*70)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {total_trainable_params:,}")
    print(f"Non-trainable params: {total_params - total_trainable_params:,}")
    print("="*70)
    
    # Calculate model size
    param_size = total_params * 4  # Assuming float32
    print(f"Model size: {param_size / 1024 / 1024:.2f} MB")
    print("="*70)