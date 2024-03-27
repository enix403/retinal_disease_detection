# Retinal Diseases Detection Project

## Overview
This project implements detections of retinal diseases using a convolutional neural network (CNN) trained on fundus image analysis of human eyes. The dataset contains labeled images of retinal scans, and the model classifies these images as either diseased or healthy.

### Model Architecture
The CNN architecture is defined in the `ConvNetwork` class. It includes the following layers:

1. **Conv2D Layers**:
   - Three convolutional layers with increasing filter sizes: 32, 64, and 128.
   - ReLU activation for non-linearity.
2. **Pooling Layers**:
   - MaxPooling layers to reduce spatial dimensions.
3. **Fully Connected Layers**:
   - Flattening of feature maps followed by dense layers with dropout to prevent overfitting.
4. **Output Layer**:
   - Linear layer with two outputs for binary classification.

### Dataset
The [Retinal Disease Classification](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification) dataset is used. Each example contains:

- An image tensor with shape `(H, W, C)`.
- A binary label indicating disease presence (`0` or `1`).

---

## Future Improvements

There are some improvements that I would like to try some day, such as:

- Implement advanced architectures such as ResNet or EfficientNet.
- Add data more augmentation techniques to improve model generalization.
- Include some evaluation metrics.