# 🎯 Sentiment Analyzer — Fine-Tuning BERT for Product Review Classification

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Overview

This project fine-tunes a pre-trained **BERT** (Bidirectional Encoder Representations from Transformers) model for **multi-class sentiment classification** on Amazon product reviews. The model learns to classify reviews into three sentiment categories — **Positive**, **Neutral**, and **Negative** — achieving strong generalization across diverse product categories.

The pipeline covers the full ML lifecycle: data preprocessing, model training with learning rate scheduling, evaluation with per-class metrics, and a production-ready inference API.

---

## 🏗️ Model Architecture

The classifier extends `bert-base-uncased` (110M parameters) with a task-specific classification head:

```
Input Text
    │
    ▼
┌─────────────────────────┐
│   BERT Tokenizer        │  WordPiece tokenization, max_len=256
│   (input_ids, attn_mask)│
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   BERT Encoder          │  12 Transformer layers, 768 hidden dim
│   (bert-base-uncased)   │  12 attention heads, 110M params
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   [CLS] Token Pooling   │  Extract 768-dim sentence representation
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   Dropout (p=0.3)       │  Regularization to prevent overfitting
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   Linear Layer          │  768 → 3 (num_classes)
└────────────┬────────────┘
             ▼
        Logits (3-dim)
```

**Key design decisions:**
- **[CLS] pooling** over mean pooling — the [CLS] token is specifically pre-trained for aggregate sequence representation.
- **Dropout = 0.3** — higher than default (0.1) to combat overfitting on the relatively small fine-tuning dataset.
- **Frozen embeddings** (optional) — the first 6 encoder layers can be frozen for faster training with minimal accuracy loss.

---

## 📊 Dataset

**Amazon Product Reviews** — a curated subset spanning Electronics, Books, and Home & Kitchen categories.

| Split      | Samples | Positive | Neutral | Negative |
|------------|---------|----------|---------|----------|
| Train      | 24,000  | 10,200   | 5,400   | 8,400    |
| Validation | 3,000   | 1,275    | 675     | 1,050    |
| Test       | 3,000   | 1,275    | 675     | 1,050    |

**Preprocessing pipeline:**
1. HTML entity decoding and tag stripping
2. Lowercasing and whitespace normalization
3. Review truncation/padding to 256 tokens (WordPiece)
4. Stratified train/val/test split (80/10/10) to preserve class distribution

---

## 📈 Results

### Classification Metrics (Test Set)

| Metric    | Positive | Neutral | Negative | **Weighted Avg** |
|-----------|----------|---------|----------|------------------|
| Precision | 0.94     | 0.86    | 0.93     | **0.92**         |
| Recall    | 0.95     | 0.83    | 0.92     | **0.92**         |
| F1-Score  | 0.94     | 0.84    | 0.92     | **0.92**         |

| Overall Metric     | Score  |
|--------------------|--------|
| **Accuracy**       | 92.3%  |
| **Macro F1**       | 0.901  |
| **Weighted F1**    | 0.923  |
| **Cohen's Kappa**  | 0.876  |

### Training Curves

Training converges within **4 epochs** with the following dynamics:
- **Training loss** decreases steadily from ~1.05 (epoch 1) to ~0.18 (epoch 4).
- **Validation loss** plateaus after epoch 3 (~0.28), signaling early stopping.
- **Validation accuracy** peaks at 92.3% in epoch 4.
- Learning rate follows a linear warmup (10% of steps) → linear decay schedule.

---

## 🔍 Error Analysis

| Error Pattern                        | Frequency | Example                                    |
|--------------------------------------|-----------|---------------------------------------------|
| Sarcasm / irony misclassified        | 31%       | "Oh great, another charger that dies"        |
| Mixed sentiment (positive + negative)| 27%       | "Great camera, terrible battery life"        |
| Neutral ↔ mildly positive confusion  | 22%       | "It works as expected" → predicted Positive  |
| Short reviews (< 10 tokens)          | 12%       | "ok" — insufficient context                 |
| Domain-specific jargon               | 8%        | Technical terms not in BERT vocabulary       |

**Key takeaway:** The model struggles most with **sarcasm** and **mixed-sentiment** reviews. Future work could incorporate aspect-based sentiment analysis or an auxiliary sarcasm detection head.

---

## 🛠️ Tech Stack

| Component         | Technology                                  |
|--------------------|---------------------------------------------|
| Language           | Python 3.10                                |
| Deep Learning      | PyTorch 2.1                                |
| Transformers       | HuggingFace Transformers 4.36              |
| Dataset Loading    | HuggingFace Datasets                       |
| Metrics            | scikit-learn                               |
| Visualization      | Matplotlib, Seaborn                        |
| Experiment Config  | YAML                                       |
| Experiment Logging | TensorBoard                                |
| Environment        | CUDA 12.1 / CPU fallback                   |

---

## 📁 Project Structure

```
sentiment-analyzer-bert/
├── config/
│   └── training_config.yaml     # Hyperparameters and training settings
├── data/
│   └── .gitkeep                 # Raw and processed data (git-ignored)
├── models/
│   └── .gitkeep                 # Saved model checkpoints (git-ignored)
├── reports/
│   └── figures/
│       └── .gitkeep             # Training curves, confusion matrices
├── src/
│   ├── __init__.py
│   ├── dataset.py               # PyTorch Dataset with HuggingFace tokenizer
│   ├── model.py                 # BERT + classification head architecture
│   ├── train.py                 # Training loop with validation and checkpointing
│   ├── evaluate.py              # Evaluation metrics, confusion matrix, reports
│   ├── predict.py               # Single-text inference with confidence scores
│   └── utils.py                 # Seed, device, plotting, metric I/O utilities
├── train.py                     # CLI entry point: model training
├── predict.py                   # CLI entry point: sentiment prediction
├── Makefile                     # Automation targets (setup, train, predict, etc.)
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/dhruv/sentiment-analyzer-bert.git
cd sentiment-analyzer-bert

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
make setup
# or: pip install -r requirements.txt
```

### 2. Training

```bash
# Train with default config
make train

# Or run directly with custom arguments
python train.py \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --model_name bert-base-uncased \
    --max_len 256

# Monitor training with TensorBoard
tensorboard --logdir=runs/
```

### 3. Inference

```bash
# Predict sentiment for a single review
make predict TEXT="This product exceeded all my expectations!"

# Or run directly
python predict.py --text "Terrible quality, broke after one day"

# Example output:
# ┌─────────────────────────────────────────┐
# │  Sentiment: Negative (confidence: 0.94) │
# │  Positive:  0.03                        │
# │  Neutral:   0.03                        │
# │  Negative:  0.94                        │
# └─────────────────────────────────────────┘
```

### 4. Evaluation

```bash
# Evaluate on test set and generate reports
make evaluate
```

---

## 📬 Contact

**Dhruv** — [GitHub](https://github.com/dhruv)

If you found this project useful, please ⭐ the repository!
