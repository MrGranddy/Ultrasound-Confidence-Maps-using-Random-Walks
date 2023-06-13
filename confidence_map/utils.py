from typing import Tuple, Optional

import numpy as np

def sub2ind(size: Tuple[int], rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Converts row and column subscripts into linear indices,
    basically the copy of the MATLAB function of the same name.
    https://www.mathworks.com/help/matlab/ref/sub2ind.html

    This function is Pythonic so the indices start at 0.

    Args:
        size Tuple[int]: Size of the matrix
        rows (np.ndarray): Row indices
        cols (np.ndarray): Column indices

    Returns:
        indices (np.ndarray): 1-D array of linear indices
    """
    indices = rows + cols * size[0]
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

    # Indices for all columns
    sc = np.arange(data.shape[1], dtype="float64")

    # SOURCE ELEMENTS - 1st matrix row
    # Indices for 1st row, it will be broadcasted with sc
    sr_up = np.array([0])
    seed = sub2ind(data.shape, sr_up, sc).astype("float64")
    seed = np.unique(seed)
    seeds = np.concatenate((seeds, seed))

    # Label 1
    label = np.ones_like(seed)
    labels = np.concatenate((labels, label))

    # SINK ELEMENTS - last image row
    if sink_mode == "all":
        sr_down = np.ones_like(sc) * (data.shape[0] - 1)
        seed = sub2ind(data.shape, sr_down, sc).astype("float64")       
    elif sink_mode == "mid":
        sc_down = np.array([data.shape[1] // 2])
        sr_down = np.ones_like(sc_down) * (data.shape[0] - 1)
        seed = sub2ind(data.shape, sr_down, sc_down).astype("float64")
    elif sink_mode == "min":
        # Find the minimum value in the last row
        min_val = np.min(data[-1, :])
        min_idxs = np.where(data[-1, :] == min_val)[0]
        sc_down = min_idxs
        sr_down = np.ones_like(sc_down) * (data.shape[0] - 1)
        seed = sub2ind(data.shape, sr_down, sc_down).astype("float64")
    elif sink_mode == "mask":
        coords = np.where(sink_mask != 0)
        sr_down = coords[0]
        sc_down = coords[1]
        seed = sub2ind(data.shape, sr_down, sc_down).astype("float64")

    seed = np.unique(seed)
    seeds = np.concatenate((seeds, seed))

    # Label 2
    label = np.ones_like(seed) * 2
    labels = np.concatenate((labels, label))

    return seeds, labels