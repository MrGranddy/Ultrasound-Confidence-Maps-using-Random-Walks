import argparse
import os
import time

import nibabel as nib

from confidence_map_3d.confidence_map_oct import ConfidenceMap

def main(args : argparse.Namespace) -> None:

    img = nib.load(args.input)
    img_data = img.get_fdata()

    # If mask is provided, use it
    if args.mask is not None:
        mask = nib.load(args.mask)
        mask_data = mask.get_fdata()

    img_data = img_data[..., 70:71]

    if args.mask is not None:
        mask_data = mask_data[..., 70:71]


    if args.mask is not None:
        # Create confidence map object
        cm = ConfidenceMap(alpha=2.0, beta=90.0, gamma=0.03, sink_mode="mask", sink_mask=mask_data)
    else:
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

    # Optional mask file (1's for source and 2's for sink)
    argparser.add_argument(
        "--mask",
        type=str,
        help="Path to mask file",
    )

    args = argparser.parse_args()

    main(args)