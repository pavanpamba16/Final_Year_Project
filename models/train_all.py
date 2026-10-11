"""
EXPLAINA AI: Unified Model Training & Evaluation Suite
Trains both:
1. MobileNetV2 Deep Vision Classifier on patient-partitioned blood smears -> models/image/malaria_mobilenet_v2.pt
2. Voting Ensemble (RF + CatBoost) on MultiCohort clinical manifestations -> models/clinical/clinical_model.pkl
"""

import os
import sys
import argparse

# Add parent directory to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from models.image_model import train_image_model
from models.clinical_model import train_clinical_models


def main():
    parser = argparse.ArgumentParser(description="EXPLAINA AI - Train All Diagnostic Models")
    parser.add_argument('--train-image', action='store_true', default=True, help="Train MobileNetV2 image model")
    parser.add_argument('--train-clinical', action='store_true', default=True, help="Train clinical Voting Ensemble")
    parser.add_argument('--epochs', type=int, default=4, help="Epochs for vision training")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for vision training")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPLAINA AI: MULTIMODAL MODEL TRAINING PIPELINE")
    print("=" * 70)

    if args.train_clinical:
        print("\n>>> STEP 1: TRAINING CLINICAL TRIAGE MODEL (VOTING ENSEMBLE)")
        clin_metrics = train_clinical_models()
        print("Clinical Training Completed. Held-out Test Metrics:", clin_metrics)

    if args.train_image:
        print("\n>>> STEP 2: TRAINING DEEP VISION MODEL (MOBILENET-V2)")
        img_metrics = train_image_model(epochs=args.epochs, batch_size=args.batch_size)
        print("Vision Training Completed. Held-out Test Metrics:", img_metrics)

    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED AND SAVED TO THE 'models/' DIRECTORY!")
    print("=" * 70)


if __name__ == '__main__':
    main()
