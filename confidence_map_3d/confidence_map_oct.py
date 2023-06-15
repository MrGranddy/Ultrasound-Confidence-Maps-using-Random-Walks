from typing import Literal, Tuple, Optional

import numpy as np
from oct2py import Oct2Py
import cv2

from scipy.sparse import csc_matrix
from scipy.signal import hilbert

from confidence_map_3d.utils import get_seed_and_labels

CONJUGATE_GRADIENT_MAX_ITERATIONS = 200
CONJUGATE_GRADIENT_TOLERANCE = 1e-6

class ConfidenceMap:
    """Confidence map computation class for RF ultrasound data"""

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 90.0,
        gamma: float = 0.05,
        mode: Literal["RF", "B"] = "B",
        sink_mode: Literal["all", "mid", "min", "mask"] = "all",
        sink_mask: Optional[np.ndarray] = None,
    ):
        """Compute the confidence map

        Args:
            precision: Precision to use. Can be 'float16', 'float32' or 'float64'
            alpha (float, optional): Alpha parameter. Defaults to 2.0.
            beta (float, optional): Beta parameter. Defaults to 90.0.
            gamma (float, optional): Gamma parameter. Defaults to 0.05.
            mode (str, optional): 'RF' or 'B' mode data. Defaults to 'B'.
            sink_mode (str, optional): Sink mode. Defaults to 'all'.
            sink_mask (np.ndarray, optional): Sink mask. Defaults to None.
        """

        # The hyperparameters for confidence map estimation
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mode = mode
        self.sink_mode = sink_mode
        self.sink_mask = sink_mask

        if self.sink_mode == "mask" and self.sink_mask is None:
            raise ValueError("Sink mask must be provided when sink mode is mask, please use 'sink_mask' argument.")

        # The precision to use for all computations
        self.eps = np.finfo("float64").eps

        # Octave instance for computing the confidence map
        self.oc = Oct2Py()

    def normalize(self, inp: np.ndarray) -> np.ndarray:
        """Normalize an array to [0, 1]"""
        return (inp - np.min(inp)) / (np.ptp(inp) + self.eps)

    def attenuation_weighting(self, A: np.ndarray, alpha: float) -> np.ndarray:
        """Compute attenuation weighting

        Args:
            A (np.ndarray): Image
            alpha: Attenuation coefficient (see publication)

        Returns:
            W (np.ndarray): Weighting expressing depth-dependent attenuation
        """

        # Create depth vector and repeat it for each column then for each stack
        Dw = np.linspace(0, 1, A.shape[0], dtype="float64")
        Dw = np.tile(Dw.reshape(-1, 1), (1, A.shape[1]))
        Dw = np.tile(Dw.reshape(-1, A.shape[1], 1), (1, A.shape[2]))

        W = 1.0 - np.exp(-alpha * Dw)  # Compute exp inline

        return W

    def confidence_laplacian(
        self, A: np.ndarray, beta: float, gamma: float
    ) -> csc_matrix:
        """Compute 6-Connected Laplacian for confidence estimation problem

        Args:
            P (np.ndarray): The index matrix of the image with boundary padding.
            A (np.ndarray): The padded image.
            beta (float): Random walks parameter that defines the sensitivity of the Gaussian weighting function.
            gamma (float): Horizontal penalty factor that adjusts the weight of horizontal edges in the Laplacian.

        Returns:
            L (csc_matrix): The 6-connected Laplacian matrix used for confidence map estimation.
        """

        m, n, h = A.shape

        idx_tensor = np.indices((m, n, h))

        # Entries vector, initially for self loops
        s = np.zeros((m * n * h), dtype="float64")
        i = np.arange(m * n * h, dtype="int64")
        j = np.arange(m * n * h, dtype="int64")

        # Vertical moves
        vertical_moving_edges = np.zeros((0), dtype="int64")

        for vertical_move in [-1, 0, 1]:
            for horizontal_move in [-1, 0, 1]:
                for stack_move in [-1, 0, 1]:

                    if vertical_move == 0 and horizontal_move == 0 and stack_move == 0:
                        continue

                    # Compute the indices of the neighboring pixels
                    neighbor_idx = np.copy(idx_tensor)
                    neighbor_idx[0] += vertical_move
                    neighbor_idx[1] += horizontal_move
                    neighbor_idx[2] += stack_move

                    # Convert into index list
                    neighbor_idx = neighbor_idx.reshape(3, -1).T
                    current_idx = idx_tensor.reshape(3, -1).T

                    # Mask out indices that are out of bounds
                    mask = (neighbor_idx[:, 0] >= 0)
                    mask &= (neighbor_idx[:, 0] < m)
                    mask &= (neighbor_idx[:, 1] >= 0)
                    mask &= (neighbor_idx[:, 1] < n)
                    mask &= (neighbor_idx[:, 2] >= 0)
                    mask &= (neighbor_idx[:, 2] < h)
                    
                    neighbor_idx = neighbor_idx[mask]
                    current_idx = current_idx[mask]

                    # Compute the intensity difference between the current pixel and its neighbors
                    neigh_values = A[neighbor_idx[:, 0], neighbor_idx[:, 1], neighbor_idx[:, 2]]
                    current_values = A[current_idx[:, 0], current_idx[:, 1], current_idx[:, 2]]

                    W = np.abs(neigh_values - current_values).reshape(-1)

                    ii = current_idx[:, 0] * n * h + current_idx[:, 1] * h + current_idx[:, 2]
                    i = np.concatenate((i, ii))
                    jj = neighbor_idx[:, 0] * n * h + neighbor_idx[:, 1] * h + neighbor_idx[:, 2]
                    j = np.concatenate((j, jj))

                    if vertical_move != 0:
                        curr_vertical_moves = s.shape[0] + np.arange(neighbor_idx.shape[0])
                        vertical_moving_edges = np.concatenate((vertical_moving_edges, curr_vertical_moves))

                    # Add the entries to the entries vector
                    s = np.concatenate((s, W))


        # Normalize weights
        s = self.normalize(s)

        # Horizontal penalty
        #s[vertical_moving_edges] += gamma

        #s[vertical_end:diagonal_end] += gamma * np.sqrt(2) # --> In the paper it is sqrt(2) since the diagonal edges are longer yet does not exist in the original code

        # Normalize differences
        s = self.normalize(s)

        # Gaussian weighting function
        s = -(
            (np.exp(-beta * s, dtype="float64")) + 1.0e-6
        )  # --> This epsilon changes results drastically default: 1.e-6

        # Create Laplacian, diagonal missing
        L = csc_matrix((s, (i, j)))

        # Reset diagonal weights to zero for summing
        # up the weighted edge degree in the next step
        L.setdiag(0)

        # Weighted edge degree
        D = np.abs(L.sum(axis=0).A)[0]

        # Finalize Laplacian by completing the diagonal
        L.setdiag(D)

        return L

    def confidence_estimation(self, A, seeds, labels, beta, gamma):
        """Compute confidence map

        Args:
            A (np.ndarray): Processed image.
            seeds (np.ndarray): Seeds for the random walks framework. These are indices of the source and sink nodes.
            labels (np.ndarray): Labels for the random walks framework. These represent the classes or groups of the seeds.
            beta: Random walks parameter that defines the sensitivity of the Gaussian weighting function.
            gamma: Horizontal penalty factor that adjusts the weight of horizontal edges in the Laplacian.

        Returns:
            map: Confidence map which shows the probability of each pixel belonging to the source or sink group.
        """

        # Laplacian
        D = self.confidence_laplacian(A, beta, gamma)

        # Select marked columns from Laplacian to create L_M and B^T
        B = D[:, seeds]

        # Select marked nodes to create B^T
        N = A.shape[0] * A.shape[1] * A.shape[2]
        i_U = np.setdiff1d(np.arange(N), seeds.astype(int))  # Index of unmarked nodes
        B = B[i_U, :]

        # Remove marked nodes from Laplacian by deleting rows and cols
        keep_indices = np.setdiff1d(np.arange(D.shape[0]), seeds)
        D = csc_matrix(D[keep_indices, :][:, keep_indices])

        # Define M matrix
        M = np.zeros((seeds.shape[0], 1), dtype="float64")
        M[:, 0] = labels == 1

        # Right-handside (-B^T*M)
        rhs = -B @ M  # type: ignore

        # Solve system exactly
        x = self.oc.mldivide(D, rhs)[:, 0]

        # Prepare output
        probabilities = np.zeros((N,), dtype="float64")
        # Probabilities for unmarked nodes
        probabilities[i_U] = x
        # Max probability for marked node
        probabilities[seeds[labels == 1].astype(int)] = 1.0

        # Final reshape with same size as input image (no padding)
        probabilities = probabilities.reshape((A.shape[0], A.shape[1], A.shape[2]))

        return probabilities

    def __call__(self, data: np.ndarray, downsample=None) -> np.ndarray:
        """Compute the confidence map

        Args:
            data (np.ndarray): RF ultrasound data (one scanline per column)

        Returns:
            map (np.ndarray): Confidence map
        """

        # Normalize data
        data = data.astype("float64")
        data = self.normalize(data)

        if self.mode == "RF":
            # MATLAB hilbert applies the Hilbert transform to columns
            data = np.abs(hilbert(data, axis=0)).astype("float64")  # type: ignore

        seeds, labels = get_seed_and_labels(data, self.sink_mode, self.sink_mask)

        # Attenuation with Beer-Lambert
        W = self.attenuation_weighting(data, self.alpha)

        # Apply weighting directly to image
        # Same as applying it individually during the formation of the
        # Laplacian
        data = data * W

        # Find condidence values
        map_ = self.confidence_estimation(data, seeds, labels, self.beta, self.gamma)

        return map_

