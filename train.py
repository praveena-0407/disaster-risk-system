"""
train.py ─ Standalone Model Training Script
============================================
Trains all three Random Forest models and saves them to disk.

Usage
─────
  # Train on synthetic data (default 8000 samples):
  python train.py

  # Train on synthetic data with custom sample count:
  python train.py --samples 15000

  # Train on your own real CSV dataset:
  python train.py --dataset data/india_weather_disaster.csv

  # Train + label a raw CSV (no risk columns) and then train:
  python train.py --raw-dataset data/raw_weather.csv

  # Specify a custom save path:
  python train.py --save models/my_model.pkl
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

from config import DEFAULT_SYNTHETIC_SAMPLES, REAL_DATASET_PATH
from data_generator import generate_training_data, load_real_dataset, label_raw_dataset
from model import DisasterRiskModel


def main():
    parser = argparse.ArgumentParser(
        description="Train Disaster Risk ML Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--samples", type=int, default=DEFAULT_SYNTHETIC_SAMPLES,
        help="Number of synthetic samples to generate (default: 8000)"
    )
    parser.add_argument(
        "--dataset", type=str, default=REAL_DATASET_PATH,
        help="Path to a labeled CSV dataset (overrides synthetic generation)"
    )
    parser.add_argument(
        "--raw-dataset", type=str, default=None,
        help="Path to an unlabeled weather CSV; labels will be auto-generated"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Override model save path"
    )
    args = parser.parse_args()

    # ── Choose data source ────────────────────────────────────────────────────
    if args.raw_dataset:
        print(f"\n[Train] Auto-labeling raw dataset: {args.raw_dataset}")
        labeled_path = args.raw_dataset.replace(".csv", "_labeled.csv")
        df = label_raw_dataset(args.raw_dataset, output_path=labeled_path)
        print(f"[Train] Labeled dataset saved to: {labeled_path}")

    elif args.dataset:
        print(f"\n[Train] Loading real dataset: {args.dataset}")
        df = load_real_dataset(args.dataset)

    else:
        print(f"\n[Train] Generating {args.samples:,} synthetic training samples...")
        df = generate_training_data(n_samples=args.samples)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[Train] Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n  Flood Risk Distribution:")
    for k, v in df["flood_risk"].value_counts().items():
        print(f"    {k:<8}: {v:>6,}  ({v/len(df)*100:.1f}%)")

    print(f"\n  Cyclone Risk Distribution:")
    for k, v in df["cyclone_risk"].value_counts().items():
        print(f"    {k:<8}: {v:>6,}  ({v/len(df)*100:.1f}%)")

    print(f"\n  Overall Risk Distribution:")
    for k, v in df["overall_risk"].value_counts().items():
        print(f"    {k:<8}: {v:>6,}  ({v/len(df)*100:.1f}%)")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[Train] Training Random Forest models...\n")
    model   = DisasterRiskModel()
    metrics = model.train(df)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = model.save(args.save)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 52)
    print("  TRAINING COMPLETE")
    print("═" * 52)
    print(f"  Flood Risk Accuracy   : {metrics['flood_accuracy']*100:.2f}%")
    print(f"  Cyclone Risk Accuracy : {metrics['cyclone_accuracy']*100:.2f}%")
    print(f"  Overall Risk Accuracy : {metrics['overall_accuracy']*100:.2f}%")
    print(f"\n  Models saved to : {save_path}")
    print("═" * 52)


if __name__ == "__main__":
    main()