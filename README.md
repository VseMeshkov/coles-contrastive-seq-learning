# CoLES: Contrastive Learning for Event Sequences

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive implementation and experimental analysis of **CoLES (Contrastive Learning for Event Sequences)** applied to age prediction from transaction sequences. This repository serves as both a practical guide and research exploration of self-supervised representation learning for temporal event data.

## 📖 About CoLES

CoLES is a self-supervised learning framework that learns meaningful representations from event sequences without requiring labeled data. It uses contrastive learning to create embeddings where similar sequences are close together and dissimilar sequences are far apart in the representation space.

**Key Resources:**
- 📄 [CoLES Paper (arXiv)](https://arxiv.org/abs/2002.08232) - Original research paper
- 🔧 [PTLS Library](https://github.com/dllllb/pytorch-lifestream) - PyTorch implementation of CoLES and other sequence models
- 📊 [Our Experiments Notebook](experiments.ipynb) - Detailed analysis and visualizations

### Why CoLES?

Traditional supervised learning on sequential data faces challenges:
- **Limited labeled data**: Labeling event sequences is expensive and time-consuming
- **Cold start problem**: New sequences have no historical labels
- **Transfer learning**: Pre-trained representations can be used across multiple downstream tasks

CoLES addresses these by:
1. **Self-supervised pre-training**: Learn from unlabeled sequences using contrastive loss
2. **Universal representations**: Single pre-trained model works for multiple prediction tasks
3. **Data efficiency**: Better performance with fewer labeled examples in downstream tasks

## 🎯 Project Goals

This repository demonstrates:
- Complete implementation of CoLES training and evaluation pipeline
- Systematic ablation studies on key hyperparameters
- Practical insights for applying CoLES to real-world sequential data
- Reproducible experiments with detailed configs and results

## 📊 Dataset: Age Prediction from Transaction Sequences

### Overview
The **Age Prediction** dataset contains anonymized transaction event sequences from bank customers. The goal is to predict customer age groups based solely on their transaction behavior patterns.

### Task Description
- **Input**: Sequential transaction data for each customer
- **Output**: Age category classification (multi-class)
- **Challenge**: Learn meaningful representations from temporal transaction patterns that correlate with age demographics

### Data Structure

Each customer sequence consists of multiple transaction events with the following features:

#### Categorical Features
- **`small_group`**: Transaction category codes (high cardinality)
  - Represents merchant categories, payment types, or transaction purposes
  - Encoded as categorical embeddings in the model
  
- **`trans_date`**: Temporal features (high cardinality)
  - Date-related information (day of week, month, etc.)
  - Captures seasonal and temporal patterns in spending behavior

#### Numerical Features
- **`amount_rur`**: Transaction amount in Russian rubles
  - Preprocessed with log transformation to handle scale
  - Reflects spending magnitude and financial behavior

#### Target Variable
- **`age`**: Customer age group (categorical)
  - Classification target for downstream task
  - Available only for supervised fine-tuning phase
  - Not used during contrastive pre-training

### Dataset Statistics
- **Number of customers**: ~9,400
- **Average sequence length**: Varies per customer
- **Transaction features**: 3 main features (2 categorical + 1 numerical)
- **Class distribution**: Multiple age categories (balanced/imbalanced - depending on your data)


## 🛠️ Key Implementation Details

### Two-Stage Pipeline

**Stage 1: Contrastive Pre-training**
- Self-supervised learning with contrastive loss
- Validation metric: `recall_top_k` (retrieval quality)
- Early stopping based on validation performance

**Stage 2: Downstream Classification**
- Extract embeddings from best checkpoint
- Train lightweight MLP classifier
- Evaluate on age prediction task


## 🔬 Experiments

All experiments, detailed analysis, and visualizations are available in **[experiments.ipynb](experiments.ipynb)**.

The notebook includes:
- **Experiment 1**: LSTM vs GRU architecture comparison
- **Experiment 2**: Hidden dimension ablation study
- **Experiment 3**: Batch size impact analysis
- **Experiment 4**: Embedding dimension exploration

Each experiment contains interactive plots, metrics tables, and comprehensive conclusions.


## 📚 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{coles2020,
title={CoLES: Contrastive Learning for Event Sequences with Self-Supervision},
author={Kolesnikov, Sergey and Trofimov, Ilya and others},
journal={arXiv preprint arXiv:2002.08232},
year={2020}
}

