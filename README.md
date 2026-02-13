# membrane_segmentation

The model architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

### Data

The original dataset was downloaded from kaggle, the ISBI data set ([http://brainiac2.mit.edu/isbi_challenge/](https://www.kaggle.com/datasets/soumikrakshit/isbi-challenge-dataset/data))


### Data augmentation

The data for training contains 30 512*512 images, and data augmentation is used for generation more training data.


### Model

A U-Net convolutional neural network is used for image segmentation.

It is trained using Binary Cross-Entropy + Dice loss and optimized with Adam (learning rate 1e-4).

Predictions are thresholded (0.4) to produce final binary segmentation masks.

### Training

The model is trained for 50 epochs resulting in excellent overlapp, dice coefficient of 0.9496.

