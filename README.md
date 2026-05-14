## MACB-DRP

Source code and data for "Multi-Level Domain Adaptation and Contrastive Domain Isolation with Bilinear Fusion for Patient Drug Response Prediction" (AAAI 2026)

## Overview

In this study, we introduce MACB-DRP, a multi-level domain adaptation framework designed to transfer pharmacogenomic knowledge from cell lines to patient drug response prediction. MACB-DRP performs domain alignment at the tissue, drug, and sample levels through three stages. Specifically, tissue-level adaptation is facilitated through tissue type classification combined with unsupervised domain adaptation. At the drug level, adversarial transfer mitigates drug-conditioned domain shifts between cell lines and patients, while bilinear fusion captures high-order gene-drug interactions. Furthermore, sample-level adaptation is performed via contrastive anchoring, which simultaneously pulls together a small number of cross-domain sample pairs with similar features and pushes apart a large set of mismatched pairs, thereby simulating intrinsic cross-domain disparities and enforcing domain isolation. This design enables MACB-DRP to hierarchically extract and align biological, pharmacological, and representational information across cell line and patient domains, providing a robust solution for drug response prediction.

<img src="https://github.com/user-attachments/assets/e1b1b2ec-cf0e-4221-b541-fd1354fe0b9e" alt="overview" width="600" />

## Installation

1. Install anaconda:
  Instructions here: https://www.anaconda.com/download/
2. pip install -r Requirements.txt
3. The data can be downloaded from here：https://drive.google.com/file/d/1azKGGPOCJNDP4vQbV1v8p_XqCrd7ekP1/view?usp=sharing
4. Run main.py

## Project Structure
```
MACB-DRP/
├── data/
│   ├── raw_dat/              # Raw datasets
│   │   ├── CCLE/             # Cell line data
│   │   ├── GDSC/             # Drug sensitivity data
│   │   └── TCGA/             # Patient tumor data
│   └── preprocessed/         # Processed features
├── logs/                     # Training logs and saved models
├── models.py                 # Neural network architectures
├── main.py                   # Main training script
├── config.py                 # Configuration parameters
├── dataload.py               # Data loading utilities
├── pretraining.py            # Pretraining scripts
├── finetune.py               # Fine-tuning scripts
├── testing.py                # Evaluation scripts
├── myloss.py                 # Custom loss functions
├── utility.py                # Helper functions
├── requirements.txt          # Core dependencies
└── train_params.json         # Training parameters
```

## Running the Code
```
python main.py
```
