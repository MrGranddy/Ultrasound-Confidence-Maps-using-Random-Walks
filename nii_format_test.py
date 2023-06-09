import argparse
import os
import time

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from confidence_map.confidence_map_scipy import ConfidenceMap as ConfidenceMap_scipy
from confidence_map.confidence_map_oct import ConfidenceMap as ConfidenceMap_oct
from confidence_map.confidence_map_cupy import ConfidenceMap as ConfidenceMap_cupy

def save_results(img, map_, output_path):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Input")

    plt.subplot(1, 2, 2)
    plt.imshow(map_, cmap="gray")
    plt.axis("off")
    plt.title("Confidence map")

    plt.savefig(output_path)
    plt.close()

def main(args : argparse.Namespace) -> None:

    # Import confidence map function from the selected backend
    if args.backend == "numpy":
        ConfidenceMap = ConfidenceMap_scipy
    elif args.backend == "octave":
        ConfidenceMap = ConfidenceMap_oct
    elif args.backend == "cupy":
        ConfidenceMap = ConfidenceMap_cupy
    else:
        # Give error message if the backend is not supported
        raise NotImplementedError(
            f'The backend "{args.backend}" is not supported.'
        )

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    img = nib.load(args.input)
    img_data = img.get_fdata()

    #label = nib.load(args.label)
    #label_data = label.get_fdata()

    # Create confidence map object
    cm = ConfidenceMap(alpha=2.0, beta=90.0, gamma=0.03)

    print("Image shape:", img_data.shape)
    if args.early_stopping is not None:
        print(f"Early stopping after {args.early_stopping} slices")

    early_stopping = args.early_stopping

    for downsample in [None]: # You can add downsampling factors here
        processing_times = []
        for i in range(min(early_stopping, img_data.shape[2])):
            print(f"Processing slice {i}...")

            start_time = time.time()
            map_ = cm(img_data[..., i])
            processing_times.append(time.time() - start_time)
            
            # Save results
            save_results(img_data[..., i], map_, os.path.join(args.output, f"{i}.png"))

        print(f"Mean and std processing time: {np.mean(processing_times)} +- {np.std(processing_times)}")
        print(f"Processing times: {processing_times}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--backend",
        type=str,
        default="octave",
        help="Backend to use. Can be 'numpy' or 'octave'",
    )
    argparser.add_argument(
        "--input",
        type=str,
        default="../data/30.nii",
        help="Input file",
    )
    argparser.add_argument(
        "--label",
        type=str,
        default="../data/30-labels.nii",
        help="Label file",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="../nii_test/",
        help="Output directory",
    )
    argparser.add_argument(
        "--early_stopping",
        type=int,
        default=100,
        help="Early stopping iterations, if you just want to test speed",
    )

    args = argparser.parse_args()

    main(args)