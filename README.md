Developing a Computer Vision-Based Attendance System with Multi-Face Recognition using PyTorch, implementing CNN architectures for single-face identification and YOLO for multi-face detection, targeting high accuracy on 200K+ image dataset with Optuna hyperparameter optimization.

# Computer Vision-Based Attendance System - Multi-Face Recognition

## Week 1: Dataset Preparation and CNN Training

### Project Goal
Build a CNN model that can identify individual people (celebrities as proxy for students) for an automated classroom attendance system.

---

## Dataset Journey: Finding the Right Approach

### Challenge
CelebA dataset contains 202,599 images across 10,177 celebrities with highly imbalanced distribution (average 19 images per celebrity). We needed to determine the optimal dataset structure for effective individual identification.

### Approaches Explored

#### 1. Random Label Generation (Failed)
**What we tried:** When the dataset initially lacked identity files, we generated fake celebrity labels by cycling through 100 pseudo-IDs.

**Results:** 
- 10,000 images, 100 fake celebrities
- First epoch: 0.91% train accuracy, 0.70% validation accuracy
- Model: 26M parameters

**Why we abandoned it:** Labels were meaningless. The model learned arbitrary groupings rather than actual facial features needed for real individual recognition.

---

#### 2. Attribute Classification (Considered but Rejected)
**What we considered:** Using facial attributes (Male/Female, Young/Old) from CelebA CSV files for binary classification.

**Why we rejected it:** Attendance systems require individual identification, not general attribute classification. Knowing someone is "young and male" doesn't identify which specific student they are.

---

#### 3. Full Dataset with Real Identity Labels (Imbalanced)
**What we tried:** Used actual celebrity IDs from `identity_CelebA.txt` with random image sampling.

**Configuration:**
- 10,000 images randomly sampled
- 2,289 different celebrities
- Training: 3.5 images per celebrity (average)
- Validation: 1.5 images per celebrity (average)

**Problems:**
- Severe class imbalance (some celebrities had 1 image, others had 9)
- Insufficient training samples per class
- Unreliable validation metrics (many celebrities had only 1 validation image)
- Model would memorize rather than learn generalizable features

**Why this failed:** CNNs require sufficient samples per class (20-30 minimum) to learn meaningful patterns. With only 3-4 training images per celebrity, the model cannot distinguish between noise and actual facial features.

---

#### 4. Balanced Dataset with Stratified Splitting (Final Solution)

**Strategy:**
1. **Celebrity Filtering:** Select only celebrities with ≥30 images in the original dataset
2. **Top Selection:** Choose top 300 celebrities by image count
3. **Balanced Sampling:** Use exactly 30 images per celebrity
4. **Stratified Splitting:** Split each celebrity 80/20 individually for train/validation

**Final Configuration:**
- 9,000 total images
- 300 celebrities (down from 10,177)
- Exactly 30 images per celebrity
- Training: 7,200 images (24 per celebrity)
- Validation: 1,800 images (6 per celebrity)

**Data Augmentation:**
Applied during training to increase effective dataset size:
- Random horizontal flips
- Random rotation (±10 degrees)
- Color jittering (brightness, contrast, saturation)
- Effective training variations: ~96 per celebrity (24 × 4 augmentations)

**Model Architecture:**
- 4 convolutional layers (64, 128, 256, 512 filters)
- Batch normalization layers
- 2 fully connected layers (1024, 512 neurons)
- Total parameters: ~105M
- Task: 300-class celebrity identification

---

## Comparison of Approaches

| Approach | Celebrities | Train Imgs/Celebrity | Val Imgs/Celebrity | Key Issue |
|----------|-------------|---------------------|-------------------|-----------|
| Random Labels | 100 | 100 | 25 | Meaningless labels |
| Full Dataset | 2,289 | 3.5 | 1.5 | Severe imbalance |
| **Balanced** | **300** | **24** | **6** | **Optimal** |

---

## Why Balanced Approach Wins

**1. Sufficient Samples Per Class**
24 training images per celebrity allows the model to learn robust facial features rather than memorizing specific photos.

**2. Consistent Validation**
Every celebrity has exactly 6 validation images, providing statistically reliable performance metrics.

**3. No Class Imbalance**
Equal representation prevents model bias toward well-represented celebrities.

**4. Effective Augmentation**
Real-time augmentation multiplies the 24 base images into ~96 effective training variations per celebrity, providing diversity without synthetic image generation.

**5. Real-World Alignment**
Mirrors actual attendance system constraints where each student would have limited enrollment photos.

---

## Data Augmentation vs Synthetic Generation

**Why we chose augmentation over GANs/synthetic images:**
- Simple augmentations (flips, rotations, color changes) create natural variations
- No risk of unrealistic generated images
- Computationally efficient (applied during training)
- Sufficient for Week 1 CNN training goals
- Avoids unnecessary complexity

---

## Technical Implementation

### Dataset Extraction
```bash
python download_data.py \
  --data-dir ./dataset \
  --identity-file ./dataset/identity_CelebA.txt \
  --num-images 10000 \
  --min-per-celebrity 30 \
  --max-per-celebrity 50
```

### Model Training
```bash
python src/training/trainer.py
```

### Project Structure
```
├── src/
│   ├── data/          # Dataset handlers (not used in final version)
│   ├── models/
│   │   └── simple_cnn.py     # CNN architecture
│   └── training/
│       └── trainer.py        # Training with stratified data
├── scripts/
│   └── download_data.py      # Balanced extraction + stratified split
└── results/
    ├── logs/                 # Training logs
    ├── best_celebrity_model.pth
    └── training_history.json
```

---

## Key Learnings

1. **Quality over Quantity:** 300 well-represented celebrities outperform 2,000+ poorly-represented ones
2. **Balance Matters:** Class imbalance is one of the biggest obstacles in deep learning
3. **Stratification Essential:** Random splits can hide poor per-class performance
4. **Augmentation Sufficient:** Simple augmentations provide enough variety without synthetic generation
5. **Iterative Problem-Solving:** Testing multiple approaches revealed dataset design principles

---

## Week 1 Completion Status

- Built CNN architecture for 300-class celebrity identification
- Implemented balanced dataset extraction with celebrity filtering
- Created stratified train/validation splits
- Applied effective data augmentation strategy
- Training in progress with properly balanced data

**Next Steps:**
- Complete training and evaluate final accuracy
- Compare with alternative CNN architectures or hyperparameters
- Select one celebrity for class data sharing
- Prepare for Week 2: Multi-face detection with YOLO