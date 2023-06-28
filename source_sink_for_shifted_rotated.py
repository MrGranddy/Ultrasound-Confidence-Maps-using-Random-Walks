import argparse
import os

import nibabel as nib
import numpy as np
import cv2

def sink_source_for_slice(slice):

        slice = slice.astype("uint8")

        # Find contours
        contours, _ = cv2.findContours((slice != 0).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the rotated bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)

        # Convert rect to box
        box = cv2.boxPoints(rect)
        box = np.int0(box)  # convert to int

        # Sort points by y-coordinate
        sorted_points = sorted(box, key=lambda point: point[1]) # type: ignore

        # The top side is formed by the first two points (smallest y)
        top_side = sorted_points[:2]

        # The bottom side is formed by the last two points (largest y)
        bottom_side = sorted_points[2:]

        # Clip sides according to image size
        top_side = np.clip(top_side, 0, slice.shape[0] - 1)
        bottom_side = np.clip(bottom_side, 0, slice.shape[0] - 1)

        # Generate mask uint8
        mask = np.zeros((slice.shape[0], slice.shape[1]), dtype=np.uint8)

        # Draw top side (source) with label 1
        cv2.line(mask, tuple(top_side[0]), tuple(top_side[1]), 1, thickness=1)

        # Draw bottom side (sink) with label 2
        cv2.line(mask, tuple(bottom_side[0]), tuple(bottom_side[1]), 2, thickness=1)

        return mask


if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(description="Source and sink generator for shifted rotated images")
    argparser.add_argument("--input", type=str, help="A 2D image or 3D nii file")
    argparser.add_argument("--output", type=str, help="The sink and source masks to be saved")

    args = argparser.parse_args()

    # Check extension
    if os.path.splitext(args.input)[1] == ".nii":

        # Read nii file
        nii_file = nib.load(args.input)
        nii_file_data = nii_file.get_fdata()

        whole_mask = np.zeros_like(nii_file_data, dtype="uint8")

        # Iterate over slices
        for i in range(nii_file_data.shape[2]):
            slice = nii_file_data[:, :, i]
            mask = sink_source_for_slice(slice)
            whole_mask[:, :, i] = mask

        # Save mask
        nib.save(nib.Nifti1Image(whole_mask, nii_file.affine, nii_file.header), args.output)

    else:

        # Read image
        nii_file_data = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

        # Generate mask
        mask = sink_source_for_slice(nii_file_data)

        # Save mask
        cv2.imwrite(args.output, mask)

