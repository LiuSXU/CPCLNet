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
   git clone https://github.com/LiuSXU/CPCLNet.git
   cd CPCLNet
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## 🗂️ Dataset
 - **[Synapse Multi-organ CT Dataset](https://www.synapse.org/Synapse:syn3193805/wiki/217790)** : The Synapse is a comprehensive collection of abdominal CT scans annotated for liver, spleen, and both kidneys, designed for multi-organ segmentation research.
 - **[CHAOS Multi-sequence MRI Dataset](https://zenodo.org/records/3431873#.Yl_9itpBxaQ)** ：The CHAOS encompasses abdominal MRI scans acquired using T1-DUAL and T2-SPIR sequences, providing detailed ground truth masks for liver, spleen, and kidneys within the training subset.
 - **[Auto PET PET/CT Dataset:](https://autopet-iii.grand-challenge.org/dataset/)** : The AutoPET is tailored for whole-body lesion segmentation, pairing PET and CT imaging modalities to capture metabolic and anatomical information.

## 🧷 Data Preprocessing
The data preprocessing pipeline in this project is adapted from:

> C. Ouyang, C. Biffi, C. Chen, T. Kart, H. F. J. M. van der Heijden, and D. Rueckert,  
> “**Self-supervision with superpixels: Training few-shot medical image segmentation without annotation**,”  
> in *Proc. ECCV*, 2020, pp. 762–780. [[Paper](https://arxiv.org/pdf/2007.09886)]

We follow their superpixel-guided self-supervised strategy to generate high-quality pseudo-masks in a fully annotation-free manner, enabling few-shot episode sampling. Key steps include:

- SLIC superpixel segmentation to produce dense region proposals.  
- Color + texture consistency-based region merging and filtering.
- Generation of foreground/background pseudo-labels for support set construction.

## 🗃️ Usage
### 1.Data Preparation
Run the preprocessing script adapted from *Ouyang et al., ECCV 2020*:
 ```bash
  python preprocess/generate_pseudo_masks.py --data_root data
   ```
Place your medical images and masks in the following structure:
 ```bash
CPCLNet/
└── data/
    ├── images/
    │   ├── CT_001.png
    │   ├── MRI_002.png
    │   └── ...
    └── masks/
        ├── CT_001.png
        ├── MRI_002.png
        └── ...
   ```
### 2.Training
 ```bash
 python train.py
   ```
Key outputs:
- outputs/models/best_model.pth → Best model (by Dice)
- outputs/visualizations/epoch_XXX.png → Per-epoch prediction
- outputs/training_log.csv → Loss & Dice curve
### 3.Evaluation
 ```bash
python evaluate.py --model_path outputs/models/best_model.pth
   ```
