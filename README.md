# Vision Transformer Architectures for Image Classification

This project is based on the research and practical implementation described in the Master's thesis titled "Архитектуре визуелних трансформера за класификацију слика" (Vision Transformer Architectures for Image Classification). The work explores and compares various vision transformer architectures with convolutional neural networks (CNNs) in the context of image classification. The main architectures studied include ViT, DeiT, and Swin Transformers.

## Project Overview

### Introduction
The project addresses the need for efficient image classification methods in today's digital age, where the volume and diversity of visual data are rapidly increasing. The thesis provides an in-depth analysis of the evolution of image classification methods, highlighting the challenges and limitations of traditional approaches.

### Objectives
The primary objective of this project is to explore and analyze the architecture of vision transformers, comparing them with conventional CNNs. The research focuses on understanding how vision transformers, particularly ViT, DeiT, and Swin, perform in image classification tasks. The analysis includes both theoretical insights and practical implementations, using a dataset of aquatic insects as a case study.

## Vision Transformer Architectures

### 1. ViT (Vision Transformer)
The Vision Transformer (ViT) is a transformer model tailored for image processing. It segments images into patches, which are then linearly embedded and fed into a transformer encoder. The core components of ViT include:
- **Patch Embedding Module:** Converts image patches into a sequence of embeddings.
- **Position Embedding:** Adds positional information to the embeddings to retain spatial relationships.
- **Transformer Encoder:** Processes the embeddings using self-attention mechanisms.

### 2. DeiT (Data-Efficient Image Transformer)
DeiT is an extension of the ViT model designed to optimize the training process by using knowledge distillation and data augmentation techniques. Key features include:
- **Knowledge Distillation:** Reduces model complexity by learning from a smaller, distilled dataset.
- **Data Augmentation and Regularization:** Enhances model performance by applying various data transformations.

### 3. Swin Transformer
The Swin Transformer introduces a hierarchical structure that processes images at multiple scales, making it suitable for a broader range of computer vision tasks, including object detection and semantic segmentation. Its components include:
- **Patch Partition and Linear Embedding:** Divides the image into patches and projects them into a lower-dimensional space.
- **Swin Transformer Block:** Uses a shifted window mechanism to enable efficient and scalable attention across the image.

## Practical Implementation

### Dataset
The practical part of the project involves applying the ViT, DeiT, and Swin architectures to a dataset of aquatic insects. The goal is to achieve optimal classification results through transfer learning.

### Transfer Learning
Transfer learning techniques are employed to adapt pre-trained models to the specific dataset. This involves fine-tuning the models on the aquatic insect dataset to improve their performance.

### ViT Implementation
In addition to applying pre-trained models, the ViT architecture is implemented from scratch to deepen the understanding of its workings. The implementation includes:
- Training the model on the dataset.
- Evaluating the model's performance in comparison to DeiT and Swin.

## Results and Analysis

The results of the experiments are thoroughly analyzed, focusing on the performance of the different architectures. The analysis highlights the concept of inductive bias, which plays a significant role in the models' ability to generalize across different datasets.

### Comparative Analysis
A detailed comparison of the ViT, DeiT, and Swin models is provided, discussing their strengths and weaknesses in various aspects of image classification.

## Conclusion
The project contributes to the understanding and advancement of image classification methods through the application of vision transformers. The findings suggest that while vision transformers offer promising results, they require further optimization to fully surpass the performance of CNNs in certain tasks.

## References
The research is supported by an extensive review of the literature, focusing on the development and application of both CNNs and transformer-based models in computer vision.

---

This README provides a comprehensive overview of the project, highlighting the key concepts, methods, and results of the work. It should serve as a useful guide for understanding the scope and achievements of the research.
