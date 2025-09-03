# TED-MKD

Structured Multi-Teacher Knowledge Distillation through Tucker-Guided Representation Alignment and Adaptive Feature Mapping

# 

### Overview

Structured Multi-Teacher Knowledge Distillation (MKD) is a novel framework designed to improve model compression and knowledge transfer from multiple teachers to a lightweight student model.



Unlike traditional single-teacher KD approaches, our method integrates Tucker decomposition with learnable convolutional regressors to achieve adaptive feature alignment. By factorizing high-dimensional teacher tensors into compact core representations, we enable semantically rich and structurally consistent supervision from multiple teachers.



We evaluate our framework on seven benchmark datasets — MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet, Oxford-IIIT Pet, and Oxford-Flowers. The results demonstrate consistent improvements over state-of-the-art KD baselines, with notable gains on challenging datasets such as CIFAR-100, Tiny ImageNet, and Oxford-Flowers.



Our approach provides a robust, scalable, and efficient solution for multi-teacher knowledge distillation, making it suitable for both simple and fine-grained visual recognition tasks.



### Features

* Data Preprocessing: Data loading and preprocessing from the MIMIC-III database.
* Modeling: Implementation of a Cross-Window Attention Model for classification.
* Knowledge Distillation: Using a teacher-student approach for enhanced performance.
* Evaluation: Metrics such as Accuracy and F1-Score to evaluate model performance.

# 

### Data

The data used in this project is obtained from the MIMIC-III Waveform Database. This database contains over 67,000 records of physiological signals, such as ECG, ABP, and PPG, collected from ICU patients.



\- Data Source: \[MIMIC-III Waveform Database v1.0 on PhysioNet](https://physionet.org/content/mimic3wdb/1.0/)

\- License: Open Database License v1.0



For more information and to access the data, visit: \[MIMIC-III Waveform Database](https://physionet.org/content/mimic3wdb/1.0/)



Please follow the instructions on the PhysioNet website to download the data.

# 

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

