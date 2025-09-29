"""
Data utilities and helper functions for CelebA dataset handling.
Includes data validation, celebrity selection, and dataset preparation utilities.
"""

import os
import pandas as pd
import numpy as np
import shutil
import json
from typing import Dict, List, Tuple, Optional
import logging
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch

logger = logging.getLogger(__name__)


def validate_dataset_structure(root_dir: str, identity_file: str) -> bool:
    """
    Validate the CelebA dataset structure and files.
    
    Args:
        root_dir (str): Root directory of CelebA dataset
        identity_file (str): Path to identity_CelebA.txt
        
    Returns:
        bool: True if dataset structure is valid
    """
    required_paths = {
        'root_dir': root_dir,
        'img_dir': os.path.join(root_dir, 'img_align_celeba'),
        'identity_file': identity_file
    }
    
    # Check if all paths exist
    for path_name, path in required_paths.items():
        if not os.path.exists(path):
            logger.error(f"Missing {path_name}: {path}")
            return False
        logger.info(f"✓ Found {path_name}: {path}")
    
    # Check identity file format
    try:
        df = pd.read_csv(identity_file, sep=' ', header=None, names=['filename', 'identity'])
        logger.info(f"✓ Identity file contains {len(df)} entries")
        
        # Check if image files exist (sample check)
        img_dir = required_paths['img_dir']
        sample_files = df['filename'].head(10).tolist()
        missing_files = []
        
        for filename in sample_files:
            img_path = os.path.join(img_dir, filename)
            if not os.path.exists(img_path):
                missing_files.append(filename)
        
        if missing_files:
            logger.warning(f"Some image files are missing: {missing_files[:5]}")
        else:
            logger.info("✓ Sample image files exist")
            
    except Exception as e:
        logger.error(f"Error validating identity file: {e}")
        return False
    
    return True


def get_celebrity_statistics(identity_file: str) -> Dict:
    """
    Get comprehensive statistics about celebrities in the dataset.
    
    Args:
        identity_file (str): Path to identity_CelebA.txt
        
    Returns:
        Dictionary containing celebrity statistics
    """
    df = pd.read_csv(identity_file, sep=' ', header=None, names=['filename', 'identity'])
    
    # Count images per celebrity
    celebrity_counts = df['identity'].value_counts()
    
    stats = {
        'total_images': len(df),
        'total_celebrities': len(celebrity_counts),
        'images_per_celebrity': {
            'min': celebrity_counts.min(),
            'max': celebrity_counts.max(),
            'mean': celebrity_counts.mean(),
            'median': celebrity_counts.median(),
            'std': celebrity_counts.std()
        },
        'celebrity_distribution': celebrity_counts.to_dict()
    }
    
    # Find celebrities with most/least images
    stats['top_10_celebrities'] = celebrity_counts.head(10).to_dict()
    stats['bottom_10_celebrities'] = celebrity_counts.tail(10).to_dict()
    
    logger.info("Celebrity Statistics:")
    logger.info(f"  Total celebrities: {stats['total_celebrities']}")
    logger.info(f"  Total images: {stats['total_images']}")
    logger.info(f"  Images per celebrity - Min: {stats['images_per_celebrity']['min']}, "
               f"Max: {stats['images_per_celebrity']['max']}, "
               f"Mean: {stats['images_per_celebrity']['mean']:.2f}")
    
    return stats


def select_balanced_celebrities(
    identity_file: str, 
    min_images: int = 20, 
    max_images: int = 200,
    target_celebrities: Optional[int] = None
) -> List[int]:
    """
    Select celebrities with balanced number of images.
    
    Args:
        identity_file (str): Path to identity_CelebA.txt
        min_images (int): Minimum images per celebrity
        max_images (int): Maximum images per celebrity
        target_celebrities (int, optional): Target number of celebrities to select
        
    Returns:
        List of selected celebrity IDs
    """
    df = pd.read_csv(identity_file, sep=' ', header=None, names=['filename', 'identity'])
    celebrity_counts = df['identity'].value_counts()
    
    # Filter celebrities by image count
    filtered_celebrities = celebrity_counts[
        (celebrity_counts >= min_images) & (celebrity_counts <= max_images)
    ]
    
    selected_ids = filtered_celebrities.index.tolist()
    
    # If target number specified, randomly sample
    if target_celebrities and len(selected_ids) > target_celebrities:
        selected_ids = np.random.choice(
            selected_ids, 
            target_celebrities, 
            replace=False
        ).tolist()
    
    logger.info(f"Selected {len(selected_ids)} celebrities with {min_images}-{max_images} images each")
    
    return selected_ids


def create_celebrity_subset_data(
    root_dir: str,
    identity_file: str,
    celebrity_id: int,
    output_dir: str,
    copy_images: bool = False
) -> Dict:
    """
    Extract data for a specific celebrity for class sharing.
    
    Args:
        root_dir (str): Root directory of CelebA dataset
        identity_file (str): Path to identity_CelebA.txt
        celebrity_id (int): ID of the celebrity to extract
        output_dir (str): Directory to save the subset data
        copy_images (bool): Whether to copy image files
        
    Returns:
        Dictionary with celebrity information
    """
    df = pd.read_csv(identity_file, sep=' ', header=None, names=['filename', 'identity'])
    celebrity_data = df[df['identity'] == celebrity_id]
    
    if len(celebrity_data) == 0:
        raise ValueError(f"Celebrity ID {celebrity_id} not found in dataset")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save celebrity file list
    celebrity_info = {
        'celebrity_id': celebrity_id,
        'num_images': len(celebrity_data),
        'image_files': celebrity_data['filename'].tolist(),
        'extraction_date': pd.Timestamp.now().isoformat()
    }
    
    # Save metadata
    with open(os.path.join(output_dir, f'celebrity_{celebrity_id}_info.json'), 'w') as f:
        json.dump(celebrity_info, f, indent=2)
    
    # Save file list
    celebrity_data[['filename']].to_csv(
        os.path.join(output_dir, f'celebrity_{celebrity_id}_files.txt'),
        header=False,
        index=False
    )
    
    # Copy images if requested
    if copy_images:
        img_output_dir = os.path.join(output_dir, 'images')
        os.makedirs(img_output_dir, exist_ok=True)
        
        img_dir = os.path.join(root_dir, 'img_align_celeba')
        
        for filename in celebrity_data['filename']:
            src_path = os.path.join(img_dir, filename)
            dst_path = os.path.join(img_output_dir, filename)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
        
        logger.info(f"Copied {len(celebrity_data)} images to {img_output_dir}")
    
    logger.info(f"Created subset for celebrity {celebrity_id} with {len(celebrity_data)} images")
    
    return celebrity_info


def analyze_image_properties(
    root_dir: str, 
    identity_file: str, 
    sample_size: int = 1000
) -> Dict:
    """
    Analyze image properties (size, format, etc.) from a sample.
    
    Args:
        root_dir (str): Root directory of CelebA dataset
        identity_file (str): Path to identity_CelebA.txt
        sample_size (int): Number of images to sample for analysis
        
    Returns:
        Dictionary with image property statistics
    """
    df = pd.read_csv(identity_file, sep=' ', header=None, names=['filename', 'identity'])
    
    # Sample images
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    img_dir = os.path.join(root_dir, 'img_align_celeba')
    
    image_sizes = []
    formats = []
    file_sizes = []
    
    for filename in sample_df['filename']:
        img_path = os.path.join(img_dir, filename)
        
        if os.path.exists(img_path):
            try:
                # Get image properties
                with Image.open(img_path) as img:
                    image_sizes.append(img.size)  # (width, height)
                    formats.append(img.format)
                
                # Get file size
                file_sizes.append(os.path.getsize(img_path))
                
            except Exception as e:
                logger.warning(f"Error analyzing {filename}: {e}")
    
    # Analyze collected data
    if image_sizes:
        widths, heights = zip(*image_sizes)
        
        stats = {
            'sample_size': len(image_sizes),
            'image_dimensions': {
                'width': {'min': min(widths), 'max': max(widths), 'mean': np.mean(widths)},
                'height': {'min': min(heights), 'max': max(heights), 'mean': np.mean(heights)},
                'unique_sizes': len(set(image_sizes)),
                'most_common_size': Counter(image_sizes).most_common(1)[0]
            },
            'file_formats': dict(Counter(formats)),
            'file_size_mb': {
                'min': min(file_sizes) / (1024*1024),
                'max': max(file_sizes) / (1024*1024),
                'mean': np.mean(file_sizes) / (1024*1024)
            }
        }
        
        logger.info("Image Properties Analysis:")
        logger.info(f"  Sample size: {stats['sample_size']}")
        logger.info(f"  Most common size: {stats['image_dimensions']['most_common_size']}")
        logger.info(f"  File formats: {stats['file_formats']}")
        logger.info(f"  Average file size: {stats['file_size_mb']['mean']:.2f} MB")
        
        return stats
    
    return {}


def plot_celebrity_distribution(
    identity_file: str, 
    top_n: int = 20, 
    save_path: Optional[str] = None
) -> None:
    """
    Plot the distribution of images per celebrity.
    
    Args:
        identity_file (str): Path to identity_CelebA.txt
        top_n (int): Number of top celebrities to show
        save_path (str, optional): Path to save the plot
    """
    df = pd.read_csv(identity_file, sep=' ', header=None, names=['filename', 'identity'])
    celebrity_counts = df['identity'].value_counts()
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Overall distribution histogram
    ax1.hist(celebrity_counts.values, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Images')
    ax1.set_ylabel('Number of Celebrities')
    ax1.set_title('Distribution of Images per Celebrity')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top N celebrities
    top_celebrities = celebrity_counts.head(top_n)
    ax2.bar(range(len(top_celebrities)), top_celebrities.values)
    ax2.set_xlabel('Celebrity Rank')
    ax2.set_ylabel('Number of Images')
    ax2.set_title(f'Top {top_n} Celebrities by Image Count')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def create_dataset_summary_report(
    root_dir: str,
    identity_file: str,
    output_file: str = "dataset_summary.json"
) -> Dict:
    """
    Create a comprehensive summary report of the dataset.
    
    Args:
        root_dir (str): Root directory of CelebA dataset
        identity_file (str): Path to identity_CelebA.txt
        output_file (str): Output file for the report
        
    Returns:
        Dictionary containing the complete report
    """
    logger.info("Generating comprehensive dataset summary...")
    
    report = {
        'dataset_path': root_dir,
        'identity_file': identity_file,
        'analysis_date': pd.Timestamp.now().isoformat(),
        'validation': validate_dataset_structure(root_dir, identity_file),
        'celebrity_stats': get_celebrity_statistics(identity_file),
        'image_properties': analyze_image_properties(root_dir, identity_file),
    }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Dataset summary saved to {output_file}")
    
    return report


def get_memory_efficient_dataloader_config(
    dataset_size: int,
    available_memory_gb: float = 8.0
) -> Dict:
    """
    Suggest memory-efficient DataLoader configuration.
    
    Args:
        dataset_size (int): Number of samples in dataset
        available_memory_gb (float): Available GPU/system memory in GB
        
    Returns:
        Dictionary with suggested DataLoader parameters
    """
    # Estimate memory usage per sample (rough estimation)
    bytes_per_sample = 224 * 224 * 3 * 4  # float32 RGB image
    gb_per_sample = bytes_per_sample / (1024**3)
    
    # Conservative memory usage (50% of available)
    usable_memory = available_memory_gb * 0.5
    
    # Calculate batch size
    max_batch_size = int(usable_memory / gb_per_sample)
    
    # Reasonable batch sizes (powers of 2)
    reasonable_batch_sizes = [16, 32, 64, 128, 256, 512]
    batch_size = max([bs for bs in reasonable_batch_sizes if bs <= max_batch_size], default=16)
    
    # Number of workers (rule of thumb: 2-4x number of GPUs)
    num_workers = min(4, os.cpu_count() or 1)
    
    config = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'prefetch_factor': 2,
        'persistent_workers': True if num_workers > 0 else False
    }
    
    logger.info(f"Suggested DataLoader config for {dataset_size} samples:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Num workers: {num_workers}")
    logger.info(f"  Estimated memory usage: {batch_size * gb_per_sample:.2f} GB")
    
    return config