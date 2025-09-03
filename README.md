# TED-MKD

The official repository of "Structured Multi-Teacher Knowledge Distillation through Tucker-Guided Representation Alignment and Adaptive Feature Mapping"

# 

### Overview

Structured Multi-Teacher Knowledge Distillation (TED-MKD) is a novel framework designed to improve model compression and knowledge transfer from multiple teachers to a lightweight student model.



Unlike traditional single-teacher KD approaches, our method integrates Tucker decomposition with learnable convolutional regressors to achieve adaptive feature alignment. By factorizing high-dimensional teacher tensors into compact core representations, we enable semantically rich and structurally consistent supervision from multiple teachers.



TED-MKD was evaluated on seven benchmark datasets MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet, Oxford-IIIT Pet, and Oxford-Flowers. The results demonstrate consistent improvements over state-of-the-art KD baselines, with notable gains on challenging datasets such as CIFAR-100, Tiny ImageNet, and Oxford-Flowers.



TED-MKD provides a robust, scalable, and efficient solution for multi-teacher knowledge distillation, making it suitable for both simple and fine-grained visual recognition tasks.



### Features

* **Multi-Teacher Supervision:** Leverages multiple high-capacity teacher models to provide diverse and complementary knowledge for guiding the student model.
* **Tucker-Guided Feature Decomposition:** Utilizes Tucker decomposition to factorize high-dimensional teacher feature tensors into compact core representations, enabling efficient cross-model supervision.
* **Adaptive Feature Alignment:** Employs learnable convolutional regressors to adaptively align student feature maps with the decomposed teacher representations, effectively handling structural mismatches.
* **Layer-Wise Knowledge Transfer:** Projects teacher knowledge onto multiple layers of the student model, capturing both low-level spatial patterns and high-level semantic information.
* **Lightweight Student Deployment:** Produces highly compressed student networks optimized for real-time inference on resource-constrained devices without sacrificing accuracy.
* **Robust and Scalable:** Demonstrates consistent performance improvements across seven benchmark datasets, ranging from simple digit recognition (MNIST) to fine-grained visual classification (Oxford-Flowers).



### Data

To comprehensively evaluate the effectiveness and generalizability of the proposed Structured Multi-Teacher Knowledge Distillation (MKD) framework, TED-MKD conducted experiments on seven publicly available benchmark datasets spanning both coarse-grained and fine-grained visual recognition tasks:



* **MNIST:** A handwritten digit classification dataset containing 60,000 training and 10,000 testing grayscale images across 10 classes.
* **Fashion-MNIST:** A dataset of 70,000 grayscale images of clothing and fashion items, also categorized into 10 classes, designed as a more challenging replacement for MNIST.
* **CIFAR-10:** Comprises 60,000 color images of size 32×32 in 10 object categories, widely used for image classification benchmarking.
* **CIFAR-100:** Similar to CIFAR-10 but with 100 fine-grained classes, making it a more challenging dataset with 600 images per class.
* **Tiny ImageNet:** A subset of ImageNet containing 100,000 images across 200 classes, with each image resized to 64×64 pixels, designed for efficient benchmarking.
* **Oxford-IIIT Pet Dataset:** Contains 7,349 images of 37 different pet breeds, presenting a fine-grained classification challenge due to significant intra-class variation.
* **Oxford-Flowers 102:** Includes 8,189 images of 102 flower categories, widely used for fine-grained visual recognition.



These datasets cover a wide range of tasks from simple digit recognition to complex fine-grained classification allowing us to evaluate both the scalability and robustness of our Tucker-guided multi-teacher distillation framework.



### Requirements

The following libraries are required:

* Python >= 3.7
* torch
* scikit-learn
* pandas
* numpy
* tqdm
* pyyaml
* wfdb

# 

##### You can install them using:

##### ```bash

##### pip install -r requirements.txt

