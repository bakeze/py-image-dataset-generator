"""
Pipeline: Data Preprocessing → Data Augmentation

Usage:
    python pipeline.py -input=path/to/raw/images -output=path/to/augmented -limit=500

Steps:
    1. Segmentation (SAM) → Denoising → Contrast enhancement → Sharpening → Padding
    2. Augmentation (rotate, blur, noise, flip) on preprocessed images
"""

import argparse
import os
import sys
import warnings

from data_preprocessing.datapreprocessing import data_preprocess
from augmentation.augmentation import DatasetGenerator
from augmentation_config import (
    DEFAULT_OPERATIONS,
    DEFAULT_ROTATE_PROBABILITY,
    DEFAULT_ROTATE_MAX_LEFT_DEGREE,
    DEFAULT_ROTATE_MAX_RIGHT_DEGREE,
    DEFAULT_BLUR_PROBABILITY,
    DEFAULT_RANDOM_NOISE_PROBABILITY,
    DEFAULT_HORIZONTAL_FLIP_PROBABILITY,
    DEFAULT_VERTICAL_FLIP_PROBABILITY,
)

warnings.filterwarnings("ignore")

DEFAULT_LIMIT = 500
DEFAULT_OUTPUT = "output"
PREPROCESSED_DIR = os.path.join("data_preprocessing", "preprocessed")
PREPROCESSED_PADDING_DIR = os.path.join(PREPROCESSED_DIR, "padding")


def run_preprocessing(input_dir: str, sam_weights: str, verbose: bool = True):
    """Run the full preprocessing pipeline on raw images."""
    print("=" * 60)
    print("  STEP 1 — Data Preprocessing")
    print("=" * 60)
    print(f"  Input directory : {input_dir}")
    print(f"  SAM weights     : {sam_weights}")
    print()

    # Ensure output sub-directories exist
    for sub in ["segmentation", "denoise", "contrast", "sharpened", "padding"]:
        os.makedirs(os.path.join(PREPROCESSED_DIR, sub), exist_ok=True)

    # Run the full preprocessing chain
    data_preprocess(input_dir)

    nb_images = len([f for f in os.listdir(PREPROCESSED_PADDING_DIR)
                     if os.path.isfile(os.path.join(PREPROCESSED_PADDING_DIR, f))])
    print(f"\n  Preprocessing complete — {nb_images} images ready in {PREPROCESSED_PADDING_DIR}")
    return PREPROCESSED_PADDING_DIR


def run_augmentation(input_dir: str, output_dir: str, limit: int):
    """Run data augmentation on preprocessed images."""
    print()
    print("=" * 60)
    print("  STEP 2 — Data Augmentation")
    print("=" * 60)
    print(f"  Input directory  : {input_dir}")
    print(f"  Output directory : {output_dir}")
    print(f"  Images to generate : {limit}")
    print()

    generator = DatasetGenerator(
        folder_path=input_dir,
        num_files=limit,
        save_to_disk=True,
        folder_destination=output_dir,
    )

    # Configure operations from augmentation_config.py
    if "rotate" in DEFAULT_OPERATIONS:
        generator.rotate(
            probability=DEFAULT_ROTATE_PROBABILITY,
            max_left_degree=DEFAULT_ROTATE_MAX_LEFT_DEGREE,
            max_right_degree=DEFAULT_ROTATE_MAX_RIGHT_DEGREE,
        )
    if "blur" in DEFAULT_OPERATIONS:
        generator.blur(probability=DEFAULT_BLUR_PROBABILITY)
    if "random_noise" in DEFAULT_OPERATIONS:
        generator.random_noise(probability=DEFAULT_RANDOM_NOISE_PROBABILITY)
    if "horizontal_flip" in DEFAULT_OPERATIONS:
        generator.horizontal_flip(probability=DEFAULT_HORIZONTAL_FLIP_PROBABILITY)
    if "vertical_flip" in DEFAULT_OPERATIONS:
        generator.vertical_flip(probability=DEFAULT_VERTICAL_FLIP_PROBABILITY)

    generator.execute()

    print(f"\n  Augmentation complete — results in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: Data Preprocessing → Data Augmentation"
    )

    parser.add_argument(
        "-input", "-i",
        required=True,
        type=str,
        help="Path to the folder containing raw images",
    )
    parser.add_argument(
        "-output", "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Destination folder for augmented images (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-limit", "-l",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of augmented images to generate (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "-sam_weights",
        type=str,
        default="sam3.pt",
        help="Path to SAM model weights (default: sam3.pt)",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        default=False,
        help="Skip preprocessing and run augmentation directly on already preprocessed images",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        default=False,
        help="Run only the preprocessing step (no augmentation)",
    )

    args = parser.parse_args()

    # ── Step 1: Preprocessing ──────────────────────────────────
    if not args.skip_preprocessing:
        preprocessed_dir = run_preprocessing(
            input_dir=args.input,
            sam_weights=args.sam_weights,
        )
    else:
        preprocessed_dir = args.input
        print(f"  Skipping preprocessing — using {preprocessed_dir} directly")

    # ── Step 2: Augmentation ───────────────────────────────────
    if not args.preprocess_only:
        run_augmentation(
            input_dir=preprocessed_dir,
            output_dir=args.output,
            limit=args.limit,
        )

    print()
    print("=" * 60)
    print("  Pipeline finished!")
    print("=" * 60)


if __name__ == "__main__":
    main()
