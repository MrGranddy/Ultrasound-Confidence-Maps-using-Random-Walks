import argparse
import os

import nibabel as nib
import numpy as np
import cv2


def shortest_distance(array, start, end):
    # Convert to numpy arrays for easier manipulation
    array = np.array(array)
    start = np.array(start)
    end = np.array(end)

    # Calculate the differences between the start and end points
    diff = end - start

    # Calculate the numerator and denominator of the distance formula
    numerator = np.abs(diff[0] * (start[1] - array[:, 1]) - (start[0] - array[:, 0]) * diff[1])
    denominator = np.sqrt(diff[0]**2 + diff[1]**2)

    # Return the shortest distance from each point in the array to the line
    if denominator == 0:
        return 1000000
    
    return numerator / denominator

def sink_source_for_slice(slice):

    slice = slice.astype("uint8")
    nonzero_mask = slice != 0

    # Find contours
    contours, _ = cv2.findContours(nonzero_mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the rotated bounding box of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)

    # Convert rect to box
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # convert to int

    # Sort points by y-coordinate
    sorted_points = sorted(box, key=lambda point: point[1])

    # The top side is formed by the first two points (smallest y)
    top_side = sorted_points[:2]

    # The bottom side is formed by the last two points (largest y)
    bottom_side = sorted_points[2:]

    # Nonzero points
    nonzero_y, nonzero_x = np.where(nonzero_mask)
    nonzero_coords = np.array(list(zip(nonzero_x, nonzero_y)))

    # Calculate the shortest distance from each nonzero point to the top and bottom sides
    top_distances = shortest_distance(nonzero_coords, top_side[0], top_side[1])
    bottom_distances = shortest_distance(nonzero_coords, bottom_side[0], bottom_side[1])

    mask = np.zeros_like(slice, dtype="uint8")

    threshold = 5
    # Set the source points to 1 which are the points closer than the threshold to the top side
    coords = nonzero_coords[np.where(top_distances < threshold)]
    mask[coords[:, 1], coords[:, 0]] = 1

    # Set the sink points to 2 which are the points closer than the threshold to the bottom side
    coords = nonzero_coords[np.where(bottom_distances < threshold)]
    mask[coords[:, 1], coords[:, 0]] = 2

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

