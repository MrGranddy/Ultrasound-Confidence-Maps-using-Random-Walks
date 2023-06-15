import argparse
import os
import time

import nibabel as nib

from confidence_map_3d.confidence_map_oct import ConfidenceMap

def main(args : argparse.Namespace) -> None:

    img = nib.load(args.input)
    img_data = img.get_fdata()

    img_data = img_data[..., :10]

    # Create confidence map object
    cm = ConfidenceMap(alpha=2.0, beta=90.0, gamma=0.03, sink_mode="all")

    out_data = cm(img_data)

    # Save results
    out_img = nib.Nifti1Image(out_data, img.affine, img.header)
    org_name = os.path.basename(args.input)
    nib.save(out_img, os.path.join( os.path.dirname(args.input), "confidence_map_3d_" + org_name))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--input",
        type=str,
        help="Path to input file",
    )


    args = argparser.parse_args()

    main(args)