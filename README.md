# CPCLNet：Advancing Cross-Modal Few-Shot Medical Image Segmentation with Clustering-Inspired Prototype Consistency Learning
## 📇Overview
This repository contains the official implementation of CPCLNet, as presented in the paper “*Advancing Cross-Modal Few-Shot Medical Image Segmentation with Clustering-Inspired Prototype Consistency Learning.*”
CPCLNet introduces a novel framework for cross-modal few-shot medical image segmentation, designed to address challenges of data scarcity and domain discrepancy among different imaging modalities such as CT, MRI, and PET. By incorporating clustering-guided prototype learning and cross-modal consistency optimization, CPCLNet achieves robust segmentation performance and enhanced generalization across unseen modalities.

<img width="1077" height="614" alt="image" src="https://github.com/user-attachments/assets/767bde72-3d0e-4a1d-95e0-cfca236d0a5f" />

## 🗄️ Introduction
 CPCLNet introduces three key modules to enhance cross-modal few-shot medical image segmentation:
 - **Cross-Modal Dynamic Alignment (CMDA) Module** : Projects support and query features into a unified latent space and adaptively aligns them through weighted regularization, enabling precise segmentation under distribution shifts.
- **Clustering-inspired Representative Prototype Descriptor (CRPD) Module**: Enforces consistency between prototypes across different modalities, improving prototype stability and cross-modal generalization.
- **RWKV-based Affinity Map Prediction (RAMP) Module** : Generates robust and discriminative prototypes by clustering foreground and background features to better capture intra-class diversity.
Together, these components allow CPCLNet to achieve robust, generalizable, and modality-consistent segmentation performance under limited supervision.

## 🗃️ Requirements
1. Clone the repository:
   ```bash
   git clone https://github.com/.git
   cd 
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## 🗂️ Dataset & Preparation
