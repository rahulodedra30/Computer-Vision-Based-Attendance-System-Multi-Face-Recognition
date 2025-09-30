Developing a Computer Vision-Based Attendance System with Multi-Face Recognition using PyTorch, implementing CNN architectures for single-face identification and YOLO for multi-face detection, targeting high accuracy on 200K+ image dataset with Optuna hyperparameter optimization.

# Computer Vision-Based Attendance System - Multi-Face Recognition

## Milestone 1: CNN for Individual Celebrity Identification

Developed a CNN-based system for automated classroom attendance using CelebA dataset (celebrities as student proxies). Goal: Build and evaluate CNN models for individual identification.

---

## Dataset Evolution

### 1. Random Labels (Failed)
- **Approach:** Generated 100 fake celebrity IDs when identity file was missing
- **Result:** 0.91% train, 0.70% val accuracy
- **Issue:** Meaningless labels, no real facial features learned

### 2. Full Dataset - Imbalanced (Failed)
- **Approach:** 10K random images from 202K dataset with real celebrity IDs
- **Result:** 2,289 celebrities, 3.5 train images/celebrity, 1.5 val images/celebrity
- **Issue:** Severe class imbalance, insufficient samples per class

### 3. Balanced + Stratified (Final Solution)
- **Approach:** Filter celebrities with ≥30 images, cap at 50 per celebrity, stratified 80/20 split
- **Result:** 333 celebrities, 24 train images/celebrity, 6 val images/celebrity
- **Success:** Every celebrity equally represented, reliable validation metrics

**Key Insight:** Quality over quantity - 300 balanced celebrities outperform 2,000+ imbalanced ones.

---

## Model Comparison

### Simple CNN (Baseline)
- **Architecture:** 4 conv layers, 2 FC layers
- **Parameters:** 104.9M
- **Dataset:** 1K images, 50 celebrities
- **Accuracy:** 4.50% validation
- **Issue:** Too large for dataset size (overfitting)

### ResNet-Style CNN (Winner)
- **Architecture:** 3 residual groups with skip connections, global pooling
- **Parameters:** 742K
- **Datasets Tested:**
  - 5K images, 166 celebrities: 9.04% accuracy
  - 10K images, 333 celebrities: **9.56% accuracy**
- **Advantage:** Residual connections + right-sized model

| Model | Params | Dataset | Classes | Val Acc | Params/Sample |
|-------|--------|---------|---------|---------|---------------|
| Simple CNN | 104.9M | 1K | 50 | 4.50% | 131,060 |
| ResNet | 0.74M | 10K | 333 | **9.56%** | 93 |

**Winner:** ResNet-Style CNN - 200x fewer parameters, 2x better accuracy, 32x better than random chance.

---

## Key Learnings

1. **Balance > Volume:** Balanced classes beat larger imbalanced datasets
2. **Stratified Splits Essential:** Ensures every celebrity in both train/val sets
3. **Right-Sized Models:** Match model capacity to dataset size
4. **Residual Connections:** Enable stable learning with limited data
5. **Augmentation Sufficient:** No need for synthetic image generation

---

## Technical Details

**Final Configuration:**
- **Dataset:** 9,990 images (7,992 train, 1,998 val)
- **Classes:** 333 celebrities (24 train, 6 val each)
- **Model:** ResNet-style CNN (742K parameters)
- **Accuracy:** 9.56% validation (32x random chance)
- **Training:** 12 epochs, Adam (lr=0.001)

**Project Structure:**
```
├── src/
│   ├── models/
│   │   ├── simple_cnn.py          # 105M param baseline
│   │   └── resnet_style_cnn.py    # 742K param winner
│   └── training/
│       └── trainer.py              # Training with logging
├── scripts/
│   └── download_data.py            # Balanced extraction + stratified split
└── results/
    ├── logs/                       # Training logs
    ├── best_celebrity_model.pth    # Best model weights
    └── training_history.json       # Metrics
```

---

## Usage

### Extract Balanced Dataset
```bash
cd scripts/
python download_data.py \
  --data-dir ./dataset \
  --identity-file ./dataset/identity_CelebA.txt \
  --num-images 20000 \
  --min-per-celebrity 30 \
  --max-per-celebrity 50
```

### Train Model
```bash
# Update dataset_dir in trainer.py to match extraction
python src/training/trainer.py
```

---

## Milestone 1 Completion

- Built and compared two CNN architectures
- Evaluated multiple dataset approaches
- Achieved 9.56% accuracy on 333-class identification
- Selected ResNet-style CNN as final model
- Ready for Week 2: Multi-face detection with YOLO

---

## Model Selection

**ResNet-Style CNN selected for:**
- Superior accuracy with 200x fewer parameters
- Stable learning progression
- Efficient deployment (742K params)
- Scalable to larger student populations
- Proven residual connection architecture