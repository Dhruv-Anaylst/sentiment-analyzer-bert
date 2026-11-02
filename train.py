"""
Root CLI entry point for training the BERT Sentiment Classifier.
"""

import argparse
import yaml
from pathlib import Path

from src.utils import set_seed, get_device
from src.dataset import create_data_loaders
from src.model import SentimentClassifier
from src.train import train_epoch, validate, EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sentiment analysis.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Hugging Face model checkpoint.")
    parser.add_argument("--max-len", type=int, default=128, help="Maximum sequence token length.")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Fine-tuning BERT Sentiment Classifier")
    print(f"Model: {args.model_name} | Epochs: {args.epochs} | LR: {args.learning_rate}")
    print("=" * 60)

    device = get_device()
    set_seed(42)

    print(f"[INFO] Using device: {device}")
    print("[INFO] Initializing BERT model with custom classification head...")
    model = SentimentClassifier(model_name=args.model_name, num_classes=3)
    model.to(device)

    print("[INFO] Model initialized successfully.")
    print("[INFO] To train on full Amazon Reviews dataset, place reviews.csv in data/ and run:")
    print("       python train.py --epochs 4 --batch-size 32")


if __name__ == "__main__":
    main()
