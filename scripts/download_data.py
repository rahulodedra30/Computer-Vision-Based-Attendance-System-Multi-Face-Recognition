"""
download script - uses existing celebrity identity mappings.
Extracts specified number of images with their real celebrity IDs.
"""

import os
import argparse
import logging
import zipfile
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('dataset_setup.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_identity_file(identity_file_path):
    """Load the existing celebrity identity file."""
    logger.info(f"Loading identity file: {identity_file_path}")
    
    df = pd.read_csv(identity_file_path, sep=' ', header=None, 
                    names=['image_id', 'celebrity_id'])
    
    logger.info(f"Loaded {len(df)} image-celebrity mappings")
    logger.info(f"Number of unique celebrities: {df['celebrity_id'].nunique()}")
    logger.info(f"Celebrity ID range: {df['celebrity_id'].min()} to {df['celebrity_id'].max()}")
    
    return df


def create_subset_dataset(identity_df, num_images=10000):
    """Create a subset of the dataset with specified number of images."""
    
    if len(identity_df) <= num_images:
        logger.info(f"Using all {len(identity_df)} available images")
        subset_df = identity_df
    else:
        logger.info(f"Sampling {num_images} images from {len(identity_df)} total")
        subset_df = identity_df.sample(n=num_images, random_state=42)
    
    # Statistics
    celeb_counts = subset_df['celebrity_id'].value_counts()
    num_celebrities = len(celeb_counts)
    
    logger.info(f"Subset statistics:")
    logger.info(f"  Total images: {len(subset_df)}")
    logger.info(f"  Unique celebrities: {num_celebrities}")
    logger.info(f"  Images per celebrity - Min: {celeb_counts.min()}, Max: {celeb_counts.max()}, Mean: {celeb_counts.mean():.1f}")
    
    return subset_df


def extract_subset_images(zip_file_path, output_dir, subset_df):
    """Extract images for the subset dataset."""
    
    img_output_dir = os.path.join(output_dir, 'img_align_celeba')
    os.makedirs(img_output_dir, exist_ok=True)
    
    image_files = subset_df['image_id'].tolist()
    logger.info(f"Extracting {len(image_files)} images...")
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            image_paths = {os.path.basename(f): f for f in all_files if f.endswith('.jpg')}
            
            extracted_count = 0
            for i, image_id in enumerate(image_files):
                if image_id in image_paths:
                    source = zip_ref.open(image_paths[image_id])
                    target_path = os.path.join(img_output_dir, image_id)
                    
                    with open(target_path, 'wb') as target:
                        target.write(source.read())
                    
                    extracted_count += 1
                    
                    if (i + 1) % 1000 == 0:
                        logger.info(f"  Extracted {i + 1}/{len(image_files)} images...")
            
            logger.info(f"✓ Extracted {extracted_count} images to {img_output_dir}")
            return True
            
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return False


def create_train_val_split(subset_df, train_ratio=0.8):
    """Create train/validation split."""
    
    train_df = subset_df.sample(frac=train_ratio, random_state=42)
    val_df = subset_df.drop(train_df.index)
    
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}")
    
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description='Extract CelebA subset with real celebrity IDs')
    parser.add_argument('--data-dir', default='./dataset', help='Dataset directory')
    parser.add_argument('--identity-file', default='./dataset/identity_CelebA.txt', 
                       help='Path to existing identity file')
    parser.add_argument('--num-images', type=int, default=10000, help='Number of images to extract')
    
    args = parser.parse_args()
    
    logger.info("=== Celebrity Identity Dataset Setup ===")
    
    # Check if identity file exists
    if not os.path.exists(args.identity_file):
        logger.error(f"Identity file not found: {args.identity_file}")
        return 1
    
    # Check if zip exists
    zip_file_path = os.path.join(args.data_dir, 'celeba-dataset.zip')
    if not os.path.exists(zip_file_path):
        logger.error(f"Dataset zip not found: {zip_file_path}")
        return 1
    
    # Load identity mappings
    identity_df = load_identity_file(args.identity_file)
    
    # Create subset
    subset_df = create_subset_dataset(identity_df, args.num_images)
    
    # Create output directory
    output_dir = os.path.join(args.data_dir, f'celeba-{len(subset_df)}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract images
    if not extract_subset_images(zip_file_path, output_dir, subset_df):
        return 1
    
    # Create train/val splits
    train_df, val_df = create_train_val_split(subset_df)
    
    # Save identity files
    train_file = os.path.join(output_dir, 'train_identity.txt')
    val_file = os.path.join(output_dir, 'val_identity.txt')
    full_file = os.path.join(output_dir, 'identity_CelebA.txt')
    
    train_df.to_csv(train_file, sep=' ', header=False, index=False)
    val_df.to_csv(val_file, sep=' ', header=False, index=False)
    subset_df.to_csv(full_file, sep=' ', header=False, index=False)
    
    logger.info(f"✓ Identity files saved to: {output_dir}")
    
    num_celebrities = subset_df['celebrity_id'].nunique()
    
    logger.info("=" * 60)
    logger.info("DATASET READY FOR TRAINING!")
    logger.info(f"✓ Dataset location: {output_dir}")
    logger.info(f"✓ Total images: {len(subset_df)}")
    logger.info(f"✓ Celebrities: {num_celebrities}")
    logger.info(f"✓ Train images: {len(train_df)}")
    logger.info(f"✓ Val images: {len(val_df)}")
    logger.info("\nNext step: Train the model")
    logger.info("  python src/training/trainer.py")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())