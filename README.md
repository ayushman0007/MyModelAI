# CIFAR-10 Classification using VGG16, PCA, Naive Bayes, and Hidden Markov Models

## Overview
This project demonstrates a machine learning pipeline for classifying images from the CIFAR-10 dataset. It leverages feature extraction using the pre-trained VGG16 model, dimensionality reduction using PCA, and classification using two models:
1. Naive Bayes Classifier
2. Hidden Markov Model (HMM)

## Dataset
The project uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 classes, such as airplanes, automobiles, birds, and more.

## Workflow
### 1. Data Preprocessing
- Load the CIFAR-10 dataset and subset it to 500 images for faster execution.
- Normalize and preprocess the images for compatibility with the VGG16 model.

### 2. Feature Extraction
- Use the pre-trained VGG16 model (without the top layer) to extract deep features from the images.
- Flatten the feature maps into 1D feature vectors.

### 3. Dimensionality Reduction
- Apply Principal Component Analysis (PCA) to reduce the feature dimensionality to 50 components.

### 4. Classification
- Train a Naive Bayes Classifier on the reduced features.
- Train a Hidden Markov Model (HMM) to classify the sequences of features.

### 5. Evaluation
- Evaluate the performance of both models using accuracy.
- Display sample predictions along with the true labels.

## Prerequisites
Ensure the following libraries are installed:
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn
- hmmlearn

To install the necessary libraries, run:
```bash
pip install tensorflow scikit-learn matplotlib hmmlearn
```

## Results
- Naive Bayes Accuracy: ~[XX]%
- HMM Accuracy: ~[YY]%

Sample predictions for the first few test images are displayed, including true labels and predictions from both models.


## Notes
- The project subsets the dataset to 500 images for faster execution. You can increase this number for better results.
- Adjust the HMM parameters (`n_components`, `n_iter`, etc.) for improved performance.

## References
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [VGG16 Model](https://keras.io/api/applications/vgg/#vgg16-function)
- [hmmlearn Documentation](https://hmmlearn.readthedocs.io/)

## License
This project is licensed under the MIT License. See `LICENSE` for details.

