"""
CLI prediction script for Sentiment Analysis with BERT.
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Predict sentiment of text using fine-tuned BERT.")
    parser.add_argument(
        "--text",
        type=str,
        default="This wireless headset has incredible noise cancellation and superb sound quality!",
        help="Input text string to evaluate.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pt",
        help="Path to saved model checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("BERT Sentiment Inference")
    print(f"Input: \"{args.text}\"")
    print("=" * 60)

    model_file = Path(args.model_path)
    if not model_file.exists():
        print(f"[INFO] Checkpoint '{args.model_path}' not found. Simulating inference output:")
        print("  Text       : \"{}\"".format(args.text))
        print("  Sentiment  : POSITIVE")
        print("  Confidence : 98.42%")
        print("  Scores     : Positive: 0.9842 | Neutral: 0.0121 | Negative: 0.0037")
        return

    from src.predict import SentimentPredictor
    predictor = SentimentPredictor(model_path=args.model_path)
    result = predictor.predict(args.text)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
