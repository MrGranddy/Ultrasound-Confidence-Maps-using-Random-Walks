from typing import Tuple, Optional

from functools import reduce

import numpy as np

def sub2ind(size: Tuple[int, int, int], row: np.ndarray, col: np.ndarray, stack:np.ndarray) -> np.ndarray:

    """Convert subscripts to linear indices

    Args:
        size (Tuple[int, int]): Size of the array
        row (np.ndarray): Row indices
        col (np.ndarray): Column indices
        stack (np.ndarray): Stack indices

    Returns:
        np.ndarray: Linear indices
    """

    row_idx = row * size[1] * size[2]
    col_idx = col * size[2]
    stack_idx = stack

    indices = row_idx + col_idx + stack_idx

    return indices

def get_seed_and_labels(data : np.ndarray, sink_mode: str = "all", sink_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Get the seed and label arrays for the max-flow algorithm

    Args:
        data: Input array
        sink_mode (str, optional): Sink mode. Defaults to 'all'.
        sink_mask (np.ndarray, optional): Sink mask. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Seed and label arrays
    """

    # Seeds and labels (boundary conditions)
    seeds = np.array([], dtype="float64")
    labels = np.array([], dtype="float64")

    # Generate a tensor with shape H, W, D, 3 where the last dimension is the indices
    idx_tensor = np.indices(data.shape)

    # SOURCE ELEMENTS - Upper plane of the image

    # Indices for the upper plane
    upper_plane = idx_tensor[:, 0, :, :]

    row_idx = upper_plane[0, ...]
    column_idx = upper_plane[1, ...]
    stack_idx = upper_plane[2, ...]

    seed = sub2ind(data.shape, row_idx, column_idx, stack_idx).astype("float64").reshape(-1)
    seeds = np.concatenate((seeds, seed))

    # Label 1
    label = np.ones_like(seed)
    labels = np.concatenate((labels, label))

    # SINK ELEMENTS - Lower plane of the image

    if sink_mode == "all":
        
        lower_plane = idx_tensor[:, -1, :, :]

        row_idx = lower_plane[0, ...]
        column_idx = lower_plane[1, ...]
        stack_idx = lower_plane[2, ...]

        seed = sub2ind(data.shape, row_idx, column_idx, stack_idx).astype("float64").reshape(-1)

    elif sink_mode == "mid":

        # Center point of the lower plane
        row_idx = np.array([data.shape[0] - 1])
        column_idx = np.array([data.shape[1] // 2])
        stack_idx = np.array([data.shape[2] // 2])

        seed = sub2ind(data.shape, row_idx, column_idx, stack_idx).astype("float64").reshape(-1)

    elif sink_mode == "min":

        # Find the minimum value in lower plane
        min_val = np.min(data[-1, ...])
        min_idxs = np.where(data[-1, ...] == min_val)

        row_idx = np.ones_like(min_idxs[0]) * (data.shape[0] - 1)
        column_idx = min_idxs[0]
        stack_idx = min_idxs[1]

        seed = sub2ind(data.shape, row_idx, column_idx, stack_idx).astype("float64").reshape(-1)

    elif sink_mode == "mask":
        coords = np.where(sink_mask != 0)

        row_idx = coords[0]
        column_idx = coords[1]
        stack_idx = coords[2]

        seed = sub2ind(data.shape, row_idx, column_idx, stack_idx).astype("float64").reshape(-1)

    seed = np.unique(seed)
    seeds = np.concatenate((seeds, seed))

    # Label 2
    label = np.ones_like(seed) * 2
    labels = np.concatenate((labels, label))

    return seeds, labels