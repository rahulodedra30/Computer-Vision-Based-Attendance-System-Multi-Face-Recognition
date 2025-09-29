"""
Balanced dataset extraction with stratified train/val splitting.
File: scripts/download_data.py
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
    """Load the celebrity identity file."""
    logger.info(f"Loading identity file: {identity_file_path}")
    
    df = pd.read_csv(identity_file_path, sep=' ', header=None, 
                    names=['image_id', 'celebrity_id'])
    
    logger.info(f"Loaded {len(df)} image-celebrity mappings")
    logger.info(f"Unique celebrities: {df['celebrity_id'].nunique()}")
    
    return df


def create_balanced_dataset(identity_df, min_images_per_celeb=30, max_images_per_celeb=100, 
                           target_total_images=50000):
    """Create balanced dataset by filtering celebrities with sufficient samples."""
    
    logger.info("Creating balanced celebrity dataset...")
    logger.info(f"  Min images per celebrity: {min_images_per_celeb}")
    logger.info(f"  Max images per celebrity: {max_images_per_celeb}")
    logger.info(f"  Target total images: {target_total_images}")
    
    # Count images per celebrity
    celeb_counts = identity_df['celebrity_id'].value_counts()
    
    # Filter celebrities with sufficient samples
    valid_celebs = celeb_counts[celeb_counts >= min_images_per_celeb].index.tolist()
    logger.info(f"Celebrities with ≥{min_images_per_celeb} images: {len(valid_celebs)}")
    
    if len(valid_celebs) == 0:
        logger.error(f"No celebrities found with ≥{min_images_per_celeb} images!")
        return None
    
    # Calculate how many celebrities we can include
    max_possible_celebs = int(target_total_images / min_images_per_celeb)
    
    if len(valid_celebs) > max_possible_celebs:
        logger.info(f"Selecting top {max_possible_celebs} celebrities with most images")
        top_celebs = celeb_counts[celeb_counts >= min_images_per_celeb].head(max_possible_celebs).index.tolist()
        valid_celebs = top_celebs
    
    logger.info(f"Selected {len(valid_celebs)} celebrities for balanced dataset")
    
    # Select images for each valid celebrity
    balanced_data = []
    
    for celeb_id in valid_celebs:
        celeb_images = identity_df[identity_df['celebrity_id'] == celeb_id]
        
        # Cap at max_images_per_celeb
        if len(celeb_images) > max_images_per_celeb:
            celeb_images = celeb_images.sample(n=max_images_per_celeb, random_state=42)
        
        balanced_data.append(celeb_images)
    
    # Combine all
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    
    # Sample down if needed
    if len(balanced_df) > target_total_images:
        logger.info(f"Sampling down to {target_total_images} images")
        
        sampled_data = []
        images_per_celeb = target_total_images // len(valid_celebs)
        
        for celeb_id in valid_celebs:
            celeb_images = balanced_df[balanced_df['celebrity_id'] == celeb_id]
            n_sample = min(len(celeb_images), images_per_celeb)
            
            if n_sample > 0:
                sampled_data.append(celeb_images.sample(n=n_sample, random_state=42))
        
        balanced_df = pd.concat(sampled_data, ignore_index=True)
    
    # Final stats
    final_counts = balanced_df['celebrity_id'].value_counts()
    
    logger.info(f"Balanced dataset created:")
    logger.info(f"  Total images: {len(balanced_df)}")
    logger.info(f"  Celebrities: {len(final_counts)}")
    logger.info(f"  Images/celebrity - Min: {final_counts.min()}, Max: {final_counts.max()}, Mean: {final_counts.mean():.1f}")
    
    return balanced_df


def create_stratified_split(balanced_df, train_ratio=0.8):
    """Create stratified split ensuring each celebrity appears in both train and val."""
    
    logger.info("Creating stratified train/validation split...")
    
    train_data = []
    val_data = []
    
    # Split each celebrity proportionally
    for celeb_id in balanced_df['celebrity_id'].unique():
        celeb_images = balanced_df[balanced_df['celebrity_id'] == celeb_id]
        
        # Shuffle
        celeb_shuffled = celeb_images.sample(frac=1, random_state=42)
        
        # Calculate split sizes
        n_total = len(celeb_shuffled)
        n_train = max(int(n_total * train_ratio), 1)  # At least 1 for training
        n_val = n_total - n_train
        
        # Ensure at least 1 in validation if possible
        if n_val == 0 and n_total > 1:
            n_train = n_total - 1
            n_val = 1
        
        # Split
        train_data.append(celeb_shuffled.iloc[:n_train])
        if n_val > 0:
            val_data.append(celeb_shuffled.iloc[n_train:])
    
    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    
    # Statistics
    train_counts = train_df['celebrity_id'].value_counts()
    val_counts = val_df['celebrity_id'].value_counts()
    
    logger.info(f"Stratified split completed:")
    logger.info(f"  Train: {len(train_df)} images, {len(train_counts)} celebrities")
    logger.info(f"  Train/celeb - Min: {train_counts.min()}, Max: {train_counts.max()}, Mean: {train_counts.mean():.1f}")
    logger.info(f"  Val: {len(val_df)} images, {len(val_counts)} celebrities")
    logger.info(f"  Val/celeb - Min: {val_counts.min()}, Max: {val_counts.max()}, Mean: {val_counts.mean():.1f}")
    
    return train_df, val_df


def extract_images(zip_file_path, output_dir, image_list):
    """Extract specified images from zip."""
    
    img_dir = os.path.join(output_dir, 'img_align_celeba')
    os.makedirs(img_dir, exist_ok=True)
    
    logger.info(f"Extracting {len(image_list)} images...")
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            image_paths = {os.path.basename(f): f for f in all_files if f.endswith('.jpg')}
            
            extracted = 0
            for i, img_id in enumerate(image_list):
                if img_id in image_paths:
                    source = zip_ref.open(image_paths[img_id])
                    target = os.path.join(img_dir, img_id)
                    
                    with open(target, 'wb') as f:
                        f.write(source.read())
                    
                    extracted += 1
                    
                    if (i + 1) % 1000 == 0:
                        logger.info(f"  Extracted {i + 1}/{len(image_list)}...")
            
            logger.info(f"✓ Extracted {extracted} images")
            return True
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Balanced CelebA dataset extraction')
    parser.add_argument('--data-dir', default='./dataset', help='Dataset directory')
    parser.add_argument('--identity-file', required=True, help='Path to identity_CelebA.txt')
    parser.add_argument('--num-images', type=int, default=10000, help='Target images')
    parser.add_argument('--min-per-celebrity', type=int, default=30, help='Min images/celebrity')
    parser.add_argument('--max-per-celebrity', type=int, default=50, help='Max images/celebrity')
    
    args = parser.parse_args()
    
    logger.info("=== Balanced Celebrity Dataset Extraction ===")
    
    # Check files
    if not os.path.exists(args.identity_file):
        logger.error(f"Identity file not found: {args.identity_file}")
        return 1
    
    zip_path = os.path.join(args.data_dir, 'celeba-dataset.zip')
    if not os.path.exists(zip_path):
        logger.error(f"Zip not found: {zip_path}")
        return 1
    
    # Load and balance
    identity_df = load_identity_file(args.identity_file)
    balanced_df = create_balanced_dataset(
        identity_df, 
        min_images_per_celeb=args.min_per_celebrity,
        max_images_per_celeb=args.max_per_celebrity,
        target_total_images=args.num_images
    )
    
    if balanced_df is None:
        return 1
    
    # Output directory
    output_dir = os.path.join(args.data_dir, f'celeba-{len(balanced_df)}-balanced')
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract images
    if not extract_images(zip_path, output_dir, balanced_df['image_id'].tolist()):
        return 1
    
    # Stratified split
    train_df, val_df = create_stratified_split(balanced_df, train_ratio=0.8)
    
    # Save files
    train_df.to_csv(os.path.join(output_dir, 'train_identity.txt'), sep=' ', header=False, index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_identity.txt'), sep=' ', header=False, index=False)
    balanced_df.to_csv(os.path.join(output_dir, 'identity_CelebA.txt'), sep=' ', header=False, index=False)
    
    logger.info("=" * 60)
    logger.info("BALANCED DATASET READY!")
    logger.info(f"✓ Location: {output_dir}")
    logger.info(f"✓ Images: {len(balanced_df)} ({len(train_df)} train, {len(val_df)} val)")
    logger.info(f"✓ Celebrities: {balanced_df['celebrity_id'].nunique()}")
    logger.info(f"✓ Stratified splits for consistent validation")
    logger.info("\nTrain: python src/training/trainer.py")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())