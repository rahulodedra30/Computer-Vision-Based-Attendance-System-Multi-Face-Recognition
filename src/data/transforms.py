"""
Data Transforms and Augmentation for CelebA Dataset
Handles image preprocessing, normalization, and augmentation strategies.
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GaussianNoise:
    """Add Gaussian noise to images."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.05):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomBrightnessContrast:
    """Random brightness and contrast adjustment."""
    
    def __init__(
        self, 
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        p: float = 0.5
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            # Adjust brightness
            brightness_factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
            
            # Adjust contrast
            contrast_factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
        
        return img


def get_base_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Get base transforms for image preprocessing.
    
    Args:
        image_size (tuple): Target image size (height, width)
        
    Returns:
        Composed transforms for basic preprocessing
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    augmentation_level: str = 'medium'
) -> transforms.Compose:
    """
    Get training transforms with augmentation.
    
    Args:
        image_size (tuple): Target image size (height, width)
        augmentation_level (str): Level of augmentation ('light', 'medium', 'heavy')
        
    Returns:
        Composed transforms for training
    """
    base_transforms = [
        transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
        transforms.RandomCrop(image_size),
    ]
    
    if augmentation_level == 'light':
        augment_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.3),
        ]
    elif augmentation_level == 'medium':
        augment_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            RandomBrightnessContrast(p=0.4),
            transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1
            ),
        ]
    elif augmentation_level == 'heavy':
        augment_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=10, 
                translate=(0.1, 0.1), 
                scale=(0.9, 1.1)
            ),
            RandomBrightnessContrast(p=0.5),
            transforms.ColorJitter(
                brightness=0.3, 
                contrast=0.3, 
                saturation=0.3, 
                hue=0.15
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
        ]
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]
    
    # Add noise as final step (applied to tensors)
    if augmentation_level in ['medium', 'heavy']:
        final_transforms.append(
            transforms.RandomApply([GaussianNoise(std=0.02)], p=0.1)
        )
    
    all_transforms = base_transforms + augment_transforms + final_transforms
    
    logger.info(f"Training transforms created with {augmentation_level} augmentation")
    return transforms.Compose(all_transforms)


def get_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Get validation transforms (no augmentation).
    
    Args:
        image_size (tuple): Target image size (height, width)
        
    Returns:
        Composed transforms for validation
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_time_augmentation_transforms(
    image_size: Tuple[int, int] = (224, 224),
    n_augmentations: int = 5
) -> list:
    """
    Get multiple transforms for test-time augmentation.
    
    Args:
        image_size (tuple): Target image size
        n_augmentations (int): Number of augmented versions to create
        
    Returns:
        List of transform compositions for TTA
    """
    tta_transforms = []
    
    # Original (no augmentation)
    tta_transforms.append(get_val_transforms(image_size))
    
    # Horizontal flip
    tta_transforms.append(transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]))
    
    # Small rotations
    for angle in [-5, 5]:
        if len(tta_transforms) >= n_augmentations:
            break
        tta_transforms.append(transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomRotation(degrees=(angle, angle)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]))
    
    # Slight brightness changes
    for brightness in [0.9, 1.1]:
        if len(tta_transforms) >= n_augmentations:
            break
        tta_transforms.append(transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=brightness),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]))
    
    return tta_transforms[:n_augmentations]


def denormalize_tensor(
    tensor: torch.Tensor, 
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor (torch.Tensor): Normalized tensor
        mean (tuple): Normalization means
        std (tuple): Normalization stds
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    if tensor.device != mean.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    return tensor * std + mean


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor to PIL Image for visualization.
    
    Args:
        tensor (torch.Tensor): Image tensor
        
    Returns:
        PIL Image
    """
    # Denormalize if normalized
    tensor = denormalize_tensor(tensor)
    
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    tensor = tensor.cpu()
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    
    array = (tensor.numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


# Transform configurations for different scenarios
TRANSFORM_CONFIGS = {
    'basic': {
        'train': lambda size: get_train_transforms(size, 'light'),
        'val': lambda size: get_val_transforms(size)
    },
    'standard': {
        'train': lambda size: get_train_transforms(size, 'medium'),
        'val': lambda size: get_val_transforms(size)
    },
    'aggressive': {
        'train': lambda size: get_train_transforms(size, 'heavy'),
        'val': lambda size: get_val_transforms(size)
    }
}


def get_transform_config(config_name: str, image_size: Tuple[int, int] = (224, 224)) -> dict:
    """
    Get a predefined transform configuration.
    
    Args:
        config_name (str): Name of configuration ('basic', 'standard', 'aggressive')
        image_size (tuple): Target image size
        
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    if config_name not in TRANSFORM_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}")
    
    config = TRANSFORM_CONFIGS[config_name]
    return {
        'train': config['train'](image_size),
        'val': config['val'](image_size)
    }