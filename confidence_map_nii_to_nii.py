import argparse
import os
import time

import numpy as np
import nibabel as nib

from utils import get_cm_backend

def main(args : argparse.Namespace) -> None:

    # Import confidence map function from the selected backend

    # Sink modes are not implemented in other than octave backend yet
    if args.backend == "octave":
        ConfidenceMap = get_cm_backend(args.backend)
    else:
        # Give error message if the backend is not supported
        raise NotImplementedError(
            f'The backend "{args.backend}" is not supported.'
        )

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    img = nib.load(args.input)
    img_data = img.get_fdata()

    # Create confidence map object
    cm = ConfidenceMap("float64", alpha=2.0, beta=90.0, gamma=0.03, sink_mode="mid")

    out_data = np.zeros(img_data.shape, dtype=np.float64)


    total_processing_time = 0
    for i in range(img_data.shape[2]):
        print(f"Processing slice {i}...")

        start_time = time.time()
        map_ = cm(img_data[..., i])
        total_processing_time += time.time() - start_time
        
        # Write results to output array
        out_data[..., i] = map_

        print(f"Slice {i} processed in {time.time() - start_time} seconds.")

    # Save results
    out_img = nib.Nifti1Image(out_data, img.affine, img.header)
    org_name = os.path.basename(args.input)
    nib.save(out_img, os.path.join(args.output, "confidence_map_" + org_name))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--backend",
        type=str,
        default="octave",
        help="Backend to use. Can be 'numpy' or 'cupy' or 'octave'",
    )
    argparser.add_argument(
        "--input",
        type=str,
        help="Path to input file",
    )
    argparser.add_argument(
        "--output",
        type=str,
        help="Output directory",
    )

    args = argparser.parse_args()

    main(args)