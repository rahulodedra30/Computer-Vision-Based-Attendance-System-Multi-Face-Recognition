Developing a Computer Vision-Based Attendance System with Multi-Face Recognition using PyTorch, implementing CNN architectures for single-face identification and YOLO for multi-face detection, targeting high accuracy on 200K+ image dataset with Optuna hyperparameter optimization.


## Week 1 Report: CNN-Based Automated Attendance System

### Project Overview

This week, our team focused on developing the foundational components for an automated classroom attendance system utilizing computer vision and deep learning techniques. The system employs Convolutional Neural Networks (CNN) to perform individual recognition from photographic inputs. To address privacy concerns associated with using actual student photographs, we implemented our solution using the CelebA dataset, where celebrity images serve as proxies for student identification.

### System Architecture and Implementation

#### Data Pipeline Development
We developed a comprehensive data management system consisting of three primary modules:

**Dataset Handler (dataset.py)**: This module manages the loading and preprocessing of celebrity images with their corresponding identity labels. We implemented label encoding functionality that converts celebrity identifiers into numerical representations suitable for machine learning algorithms. The system incorporates automatic train-validation splitting with an 80/20 ratio as specified in the project requirements.

**Image Processing Module (transforms.py)**: We designed a robust image preprocessing pipeline that standardizes input images to 224x224 pixel resolution and applies data augmentation techniques. The augmentation strategies include rotation, brightness adjustment, and horizontal flipping to enhance model generalization and prevent overfitting on specific image characteristics.

**Utility Functions (utils.py)**: We implemented supporting functions for dataset validation, statistical analysis of celebrity distribution, and extraction capabilities for individual celebrity datasets to facilitate collaboration with other project groups.

#### CNN Model Architecture
Our team designed a CNN architecture optimized for multi-class classification with the following specifications:
- Four convolutional layers with filter sizes of 32, 64, 128, and 256 respectively
- Max pooling operations following each convolutional layer for dimensionality reduction
- Two fully connected layers with 512 neurons and final classification layer
- Dropout regularization (p=0.5) to prevent overfitting
- Total trainable parameters: 26,130,340

The model accepts 224x224 RGB images as input and produces classification predictions across 100 distinct celebrity classes.

#### Training Implementation
We developed a training framework incorporating:
- Mini-batch processing with batch size of 16 images
- Adam optimizer with learning rate of 0.001
- Cross-entropy loss function for multi-class classification
- Performance tracking for both training and validation metrics
- Model checkpointing based on validation accuracy improvements

### Dataset Configuration and Challenges

The primary technical challenges encountered this week were related to CelebA dataset preparation and system integration:

1. **Metadata Extraction Issues**: The downloaded dataset archive contained only image files without the corresponding identity mapping files required for supervised learning.
2. **Storage Constraints**: We encountered disk space limitations on the university computing cluster that necessitated working with dataset subsets.
3. **Path Configuration**: We resolved multiple file path dependency issues across different system directories and modules.

We addressed these challenges through:
- Development of custom metadata generation scripts that create identity mappings from image filenames
- Implementation of selective extraction capabilities to work with 10,000 image subsets instead of the complete 200K image dataset
- Creation of a flexible download system supporting configurable dataset sizes

### Model Training
To recreate the trained model:
1. Run the data setup: `python scripts/download_data.py --num-images <num of images to extract from kaggle>`
2. Train the model: `python src/training/trainer.py`
3. Model will be saved to `results/best_model.pth`

### Experimental Results

#### Model Architecture

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 224, 224]             896
         MaxPool2d-2         [-1, 32, 112, 112]               0
            Conv2d-3         [-1, 64, 112, 112]          18,496
         MaxPool2d-4           [-1, 64, 56, 56]               0
            Conv2d-5          [-1, 128, 56, 56]          73,856
         MaxPool2d-6          [-1, 128, 28, 28]               0
            Conv2d-7          [-1, 256, 28, 28]         295,168
         MaxPool2d-8          [-1, 256, 14, 14]               0
            Linear-9                  [-1, 512]      25,690,624
          Dropout-10                  [-1, 512]               0
           Linear-11                  [-1, 100]          51,300
================================================================
Total params: 26,130,340
Trainable params: 26,130,340
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 28.72
Params size (MB): 99.68
Estimated Total Size (MB): 128.97
----------------------------------------------------------------

**Model Performance:**
- Best validation accuracy: 80.0%
- Model saved to: results/best_model.pth


### Technical Implementation Specifications

**Project Structure:**
```
Computer-Vision-Based-Attendance-System-Multi-Face-Recognition/
├── src/
│   ├── data/          # Data pipeline components
│   ├── models/        # CNN architecture definitions
│   └── training/      # Training algorithms and utilities
├── scripts/           # Dataset management and setup utilities
└── results/           # Model outputs and performance logs
```

**Technology Stack:**
- PyTorch framework for deep learning implementation
- torchvision for computer vision transformations
- pandas for data manipulation and analysis
- PIL (Python Imaging Library) for image processing
- Kaggle API for dataset acquisition

### Technical Issues and Resolutions

1. **GPU Compatibility**: We encountered CUDA compatibility issues with the Tesla P100 GPU and current PyTorch versions, requiring migration to CPU-based training.

3. **Dataset Structure Complexity**: The CelebA archive contains nested directory structures that required careful path handling and extraction logic.

4. **Memory Management**: We optimized system resource usage through reduced batch sizes and data loader worker processes to accommodate system constraints.


### Analysis and Conclusions

This week's development established a solid foundation for the automated attendance system project. While dataset preparation required more time investment than initially anticipated, we successfully created a robust and modular data processing system that will facilitate future development phases. Our approach of building reusable, well-documented components will enable efficient experimentation with model architectures and training strategies in subsequent weeks.

The experience of working within computational constraints reinforced the importance of developing adaptable solutions that can function effectively across different hardware configurations and resource limitations. The modular design approach we adopted will prove valuable as we progress toward the multi-face detection and tracking components required for a complete attendance system implementation.