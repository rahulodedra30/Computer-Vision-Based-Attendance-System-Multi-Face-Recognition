"""
CelebA Dataset Handler for Attendance System
Handles loading, preprocessing, and splitting of CelebA dataset for face recognition.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class CelebADataset(Dataset):
    """
    CelebA Dataset class for loading celebrity images and their identity labels.
    
    Args:
        root_dir (str): Root directory containing the CelebA dataset
        identity_file (str): Path to identity_CelebA.txt file
        transform (callable, optional): Optional transform to be applied on images
        subset_ids (list, optional): List of celebrity IDs to include (for filtering)
    """
    
    def __init__(
        self, 
        root_dir: str,
        identity_file: str,
        transform: Optional[callable] = None,
        subset_ids: Optional[List[int]] = None
    ):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')
        self.transform = transform
        
        # Load identity mappings
        self.identity_df = self._load_identity_file(identity_file)
        
        # Filter by subset if provided
        if subset_ids is not None:
            self.identity_df = self.identity_df[
                self.identity_df['identity'].isin(subset_ids)
            ]
            logger.info(f"Filtered dataset to {len(subset_ids)} celebrities")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.identity_df['label'] = self.label_encoder.fit_transform(
            self.identity_df['identity']
        )
        
        # Create mappings
        self.num_classes = len(self.label_encoder.classes_)
        self.id_to_name = self._create_id_mappings()
        
        logger.info(f"Dataset initialized with {len(self.identity_df)} images")
        logger.info(f"Number of unique celebrities: {self.num_classes}")
        
    def _load_identity_file(self, identity_file: str) -> pd.DataFrame:
        """Load and parse the identity_CelebA.txt file."""
        try:
            df = pd.read_csv(
                identity_file, 
                sep=' ', 
                header=None, 
                names=['filename', 'identity']
            )
            logger.info(f"Loaded {len(df)} identity mappings")
            return df
        except Exception as e:
            logger.error(f"Error loading identity file: {e}")
            raise
    
    def _create_id_mappings(self) -> Dict[int, str]:
        """Create mapping from encoded labels to original celebrity IDs."""
        return dict(zip(
            self.label_encoder.transform(self.label_encoder.classes_),
            self.label_encoder.classes_
        ))
    
    def __len__(self) -> int:
        return len(self.identity_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            tuple: (image_tensor, label) where label is the encoded celebrity ID
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image filename and label
        row = self.identity_df.iloc[idx]
        img_name = row['filename']
        label = row['label']
        
        # Load image
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (218, 178), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of samples per class."""
        return self.identity_df['label'].value_counts().to_dict()
    
    def get_celebrity_name(self, encoded_label: int) -> int:
        """Get original celebrity ID from encoded label."""
        return self.id_to_name.get(encoded_label, -1)
    
    def sample_celebrity_images(self, celebrity_id: int, n_samples: int = 10) -> List[str]:
        """
        Sample n images for a specific celebrity.
        
        Args:
            celebrity_id (int): Original celebrity ID
            n_samples (int): Number of samples to return
            
        Returns:
            List of image filenames for the celebrity
        """
        celebrity_images = self.identity_df[
            self.identity_df['identity'] == celebrity_id
        ]['filename'].tolist()
        
        if len(celebrity_images) < n_samples:
            logger.warning(
                f"Celebrity {celebrity_id} has only {len(celebrity_images)} images, "
                f"requested {n_samples}"
            )
            return celebrity_images
        
        return np.random.choice(celebrity_images, n_samples, replace=False).tolist()


def create_data_splits(
    dataset: CelebADataset, 
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Create train/validation splits from the dataset.
    
    Args:
        dataset (CelebADataset): The complete dataset
        train_ratio (float): Ratio of training data (0.8 = 80%)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    torch.manual_seed(random_seed)
    
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )
    
    logger.info(f"Dataset split - Train: {train_size}, Validation: {val_size}")
    
    return train_dataset, val_dataset


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to pin memory for GPU training
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(f"DataLoaders created - Batch size: {batch_size}")
    
    return train_loader, val_loader


def analyze_dataset_statistics(dataset: CelebADataset) -> Dict:
    """
    Analyze and return dataset statistics.
    
    Args:
        dataset (CelebADataset): Dataset to analyze
        
    Returns:
        Dictionary containing dataset statistics
    """
    class_dist = dataset.get_class_distribution()
    
    stats = {
        'total_images': len(dataset),
        'num_classes': dataset.num_classes,
        'min_samples_per_class': min(class_dist.values()),
        'max_samples_per_class': max(class_dist.values()),
        'avg_samples_per_class': np.mean(list(class_dist.values())),
        'class_distribution': class_dist
    }
    
    logger.info("Dataset Statistics:")
    logger.info(f"  Total images: {stats['total_images']}")
    logger.info(f"  Number of classes: {stats['num_classes']}")
    logger.info(f"  Min samples per class: {stats['min_samples_per_class']}")
    logger.info(f"  Max samples per class: {stats['max_samples_per_class']}")
    logger.info(f"  Avg samples per class: {stats['avg_samples_per_class']:.2f}")
    
    return stats