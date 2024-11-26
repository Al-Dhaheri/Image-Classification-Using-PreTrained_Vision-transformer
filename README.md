# Vision Transformer (ViT) for Image Classification

This project focuses on utilizing a pre-trained Vision Transformer (ViT) model to classify images from a custom dataset. The ViT model is fine-tuned using transfer learning to recognize various classes, applying advanced techniques like data augmentation to improve the accuracy of the model.

## Overview

The Vision Transformer (ViT) is a model architecture that applies Transformer-based deep learning to image classification tasks. This project makes use of ViT with pre-trained weights, fine-tuning it to classify specific types of image defects.

### Key Features:

- **Data Augmentation**: Extensive use of data augmentation techniques, such as random rotations, horizontal/vertical flips, shear, and translation, to enhance the dataset.
- **Transfer Learning**: Fine-tuned a pre-trained ViT model to classify image defects across six categories: Crazing, Inclusion, Patches, Pitted, Rolled, and Scratches.
- **Post-Training Evaluation**: Comprehensive evaluation of the model performance, including confusion matrix, class-wise AUC scores, overall model metrics such as accuracy, sensitivity, specificity, and error rate.
- **Prediction Visualization**: Visual representation of predictions on the original non-augmented dataset.

## Dataset

The dataset consists of images categorized into six classes of defects:

- **Crazing**
- **Inclusion**
- **Patches**
- **Pitted**
- **Rolled**
- **Scratches**

The images were divided into training and test datasets, with augmentation applied to the training dataset to improve model generalization.

## Training

### Model Architecture

- **Model**: Vision Transformer (ViT-B-16)
- **Pre-trained Weights**: The model starts with pre-trained weights from the ImageNet dataset.
- **Modified Classifier Head**: The classifier head was modified to classify six different defect types.
- **Training Configuration**:
  - **Optimizer**: Adam Optimizer with a learning rate of `3e-3` and weight decay of `0.01`.
  - **Loss Function**: Cross Entropy Loss.
  - **Epochs**: The model was trained for 50 epochs.

### Data Augmentation Techniques

The following data augmentation techniques were used for training:

- **Random Rotation**: Images were randomly rotated within the range [-45, 45] degrees.
- **Horizontal and Vertical Flips**: Images were flipped horizontally and vertically to increase variability.
- **Affine Transformations**: Shear and translation were applied to further generalize the model.

### Results

- **Accuracy**: The model achieved an accuracy of **99.72%** on the test dataset.
- **AUC (Average)**: The average AUC score across all classes was **1.00**.
- **Sensitivity (SE %)**: **99.73%**
- **Specificity (SP %)**: **99.73%**
- **Error Rate (ER %)**: **0.28%**

## Evaluation

After training, the model was evaluated based on the following metrics:

- **Confusion Matrix**: Displays the number of correct and incorrect predictions for each class, with both counts and percentages.
- **Classification Report**: Provides precision, recall (sensitivity), F1-score, and support for each class.
- **ROC Curves**: One-vs-Rest ROC curve was plotted for each class, displaying AUC values for individual categories.

### Model Performance Metrics

- **Model AUC**: **1.00**
- **Accuracy (AC %)**: **99.72%**
- **Sensitivity (SE %)**: **99.73%**
- **Specificity (SP %)**: **99.73%**
- **Error Rate (ER %)**: **0.28%**

### Class-wise Metrics

Each class was evaluated separately for AUC, accuracy, sensitivity, and specificity. A visual representation of the **ROC curve** for each class has been provided.

## Usage Instructions

### Prerequisites

- **Python 3.7+**
- **PyTorch** and **Torchvision** for model implementation.
- **Google Colab** or similar GPU-enabled environment is recommended for training.
- **Torchinfo** for model summary visualization.

### Running the Project

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/username/vit-image-classification.git
   cd vit-image-classification
   ```

2. **Install Requirements**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Training the Model**: You can run the training script with the following command:

   ```bash
   python train_using_pretrained_model_image_classifier.py
   ```

4. **Evaluation**: After training, the model's metrics and visualizations are automatically generated, including plots for loss, accuracy, confusion matrix, and ROC curves.

## Project Structure

- `train_using_pretrained_model_image_classifier.py`: Main script to train the ViT model on the custom dataset.
- `helper_functions.py`: Utility functions for model evaluation, plotting, and performance analysis.
- `saved_models/`: Directory containing the trained models.

## Results

The pre-trained Vision Transformer (ViT) model, after fine-tuning, exhibited strong performance in identifying defects in the custom dataset. The model achieved high accuracy, AUC, sensitivity, and specificity, making it suitable for practical use in automated defect detection.

## Future Work

- **Expand Dataset**: Increase the diversity and number of images to further improve model robustness.
- **Experiment with Different Architectures**: Test the efficiency of newer transformer architectures on the defect classification problem.
- **Hyperparameter Tuning**: Explore different learning rates, optimizers, and data augmentation settings.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project used the Vision Transformer implementation provided by PyTorch's `torchvision` library.
- Thanks to the authors of the ViT model for pioneering Transformer-based models in computer vision.

---

Feel free to fork this project, contribute, and share your insights!

