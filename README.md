# Deep Learning for Anti-Cancer Drug Response Prediction

## Overview
This repository implements a deep learning framework for predicting anti-cancer drug response. The model leverages Variational AutoEncoders (VAEs) to preprocess and encode three input feature sets:
1. Cancer cell line’s mutation profile.
2. Cancer cell line’s gene expression profile.
3. Anti-cancer drug molecular fingerprints.

The encoded features are then used as inputs to a supervised deep neural network to predict the IC50 value, which measures drug sensitivity.

## Features
- **Variational AutoEncoders (VAEs)**:
  - Mutation VAE: Encodes mutation profiles with a feed-forward neural network.
  - Expression VAE: Encodes gene expression profiles.
  - Molecule VAE: Utilizes a Junction Tree VAE for molecular graph encoding.

- **Drug Response Prediction Model**:
  - Combines encoded features from VAEs.
  - Maps input features to IC50 values using feed-forward neural networks.

- **Model Evaluation**:
  - Pearson Correlation (R²) as the primary evaluation metric.
  - Comparison of baseline models (Linear Regression and SVM) with deep learning models.

## Data
The dataset includes:
- **Gene Expression Data**: From Cancer Cell Line Encyclopedia (CCLE) and TCGA databases.
- **Mutation Profiles**: Binary mutation matrices from CCLE and TCGA.
- **Drug Sensitivity Data**: IC50 values from the Genomics of Drug Sensitivity in Cancer (GDSC) project.
- **Molecular Fingerprints**: Represented using SMILES strings from the PubChem database.

## Results
The following table compares the performance of different models using R² as the evaluation metric:

| Model                       | R²    |
|-----------------------------|-------|
| Linear Regression (LR)      | 0.57  |
| Support Vector Machine (SVM)| 0.63  |
| Deep Neural Network (DNN_i) | 0.71  |
| DNN_ii                      | 0.79  |
| DNN_iii                     | 0.81  |
| CDRScan                     | 0.84  |

## Future Work
- Extend the model for drug discovery to generate new molecules using generative models.
- Integrate convolutional neural networks to enhance prediction accuracy.
- Explore drug combination effects for multi-drug treatments.
- Incorporate additional datasets to improve generalizability.
- Test the model's scalability on large-scale datasets across different cancer types.

## References
1. Chang et al. (2018). *Cancer drug response profile scan (CDRScan)*. Scientific Reports. [Link](https://www.nature.com/articles/s41598-018-29294-6)
2. Chiu et al. (2018). *Predicting drug response of tumors from integrated genomic profiles*. arXiv. [Link](https://arxiv.org/abs/1806.10489)
3. Jin et al. (2018). *Junction Tree Variational AutoEncoder for molecular graph generation*. arXiv. [Link](https://arxiv.org/abs/1802.04364)
