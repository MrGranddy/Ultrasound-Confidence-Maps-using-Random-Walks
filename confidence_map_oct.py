from typing import Literal, Tuple, Optional

import numpy as np
import cv2

from scipy.sparse import csc_matrix
from scipy.signal import hilbert

from oct2py import Oct2Py

CONJUGATE_GRADIENT_MAX_ITERATIONS = 200
CONJUGATE_GRADIENT_TOLERANCE = 1e-6

class ConfidenceMap:
    """Confidence map computation class for RF ultrasound data"""

    def __init__(
        self,
        precision: Literal["float16", "float32", "float64"] = "float32",
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
        self.precision = precision
        self.eps = np.finfo(self.precision).eps

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

        # Create depth vector and repeat it for each column
        Dw = np.linspace(0, 1, A.shape[0], dtype=self.precision)
        Dw = np.tile(Dw.reshape(-1, 1), (1, A.shape[1]))

        W = 1.0 - np.exp(-alpha * Dw)  # Compute exp inline

        return W

    def confidence_laplacian(
        self, P: np.ndarray, A: np.ndarray, beta: float, gamma: float
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

        m, _ = P.shape

        P = P.T.flatten()
        A = A.T.flatten()

        p = np.where(P > 0)[0]

        i = P[p] - 1  # Index vector
        j = P[p] - 1  # Index vector
        # Entries vector, initially for diagonal
        s = np.zeros_like(p, dtype=self.precision)

        vl = 0  # Vertical edges length

        edge_templates = [
            -1,  # Vertical edges
            1,
            m - 1,  # Diagonal edges
            m + 1,
            -m - 1,
            -m + 1,
            m,  # Horizontal edges
            -m,
        ]

        vertical_end = None
        diagonal_end = None

        for iter_idx, k in enumerate(edge_templates):

            Q = P[p + k]

            q = np.where(Q > 0)[0]

            ii = P[p[q]] - 1
            i = np.concatenate((i, ii))
            jj = Q[q] - 1
            j = np.concatenate((j, jj))
            W = np.abs(A[p[ii]] - A[p[jj]])  # Intensity derived weight
            s = np.concatenate((s, W))

            if iter_idx == 1:
                vertical_end = s.shape[0]  # Vertical edges length
            elif iter_idx == 5:
                diagonal_end = s.shape[0]  # Diagonal edges length

        # Normalize weights
        s = self.normalize(s)

        # Horizontal penalty
        s[:vertical_end] += gamma
        #s[vertical_end:diagonal_end] += gamma * np.sqrt(2) # --> In the paper it is sqrt(2) since the diagonal edges are longer yet does not exist in the original code

        # Normalize differences
        s = self.normalize(s)

        # Gaussian weighting function
        s = -(
            (np.exp(-beta * s, dtype=self.precision)) + 1.0e-6
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

        # Index matrix with boundary padding
        G = np.arange(1, A.shape[0] * A.shape[1] + 1).reshape(A.shape[1], A.shape[0]).T
        pad = 1

        G = np.pad(G, (pad, pad), "constant", constant_values=(0, 0))
        B = np.pad(A, (pad, pad), "constant", constant_values=(0, 0))

        # Laplacian
        D = self.confidence_laplacian(G, B, beta, gamma)

        # Select marked columns from Laplacian to create L_M and B^T
        B = D[:, seeds]

        # Select marked nodes to create B^T
        N = np.sum(G > 0).item()
        i_U = np.setdiff1d(np.arange(N), seeds.astype(int))  # Index of unmarked nodes
        B = B[i_U, :]

        # Remove marked nodes from Laplacian by deleting rows and cols
        keep_indices = np.setdiff1d(np.arange(D.shape[0]), seeds)
        D = csc_matrix(D[keep_indices, :][:, keep_indices])

        # Define M matrix
        M = np.zeros((seeds.shape[0], 1), dtype=self.precision)
        M[:, 0] = labels == 1

        # Right-handside (-B^T*M)
        rhs = -B @ M  # type: ignore

        # Solve system exactly
        x = self.oc.mldivide(D, rhs)[:, 0]

        # Prepare output
        probabilities = np.zeros((N,), dtype=self.precision)
        # Probabilities for unmarked nodes
        probabilities[i_U] = x
        # Max probability for marked node
        probabilities[seeds[labels == 1].astype(int)] = 1.0

        # Final reshape with same size as input image (no padding)
        probabilities = probabilities.reshape((A.shape[1], A.shape[0])).T

        return probabilities

    def sub2ind(
        self, size: Tuple[int], rows: np.ndarray, cols: np.ndarray
    ) -> np.ndarray:
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

    def __call__(self, data: np.ndarray, downsample=None) -> np.ndarray:
        """Compute the confidence map

        Args:
            data (np.ndarray): RF ultrasound data (one scanline per column)

        Returns:
            map (np.ndarray): Confidence map
        """

        # Normalize data
        data = data.astype(self.precision)
        data = self.normalize(data)

        if self.mode == "RF":
            # MATLAB hilbert applies the Hilbert transform to columns
            data = np.abs(hilbert(data, axis=0)).astype(self.precision)  # type: ignore

        org_H, org_W = data.shape
        if downsample is not None:
            data = cv2.resize(data, (org_W // downsample, org_H // downsample), interpolation=cv2.INTER_CUBIC)

        # Seeds and labels (boundary conditions)
        seeds = np.array([], dtype=self.precision)
        labels = np.array([], dtype=self.precision)

        # Indices for all columns
        sc = np.arange(data.shape[1], dtype=self.precision)

        # SOURCE ELEMENTS - 1st matrix row
        # Indices for 1st row, it will be broadcasted with sc
        sr_up = np.array([0])
        seed = self.sub2ind(data.shape, sr_up, sc).astype(self.precision)
        seed = np.unique(seed)
        seeds = np.concatenate((seeds, seed))

        # Label 1
        label = np.ones_like(seed)
        labels = np.concatenate((labels, label))

        # SINK ELEMENTS - last image row
        if self.sink_mode == "all":
            sr_down = np.ones_like(sc) * (data.shape[0] - 1)
            seed = self.sub2ind(data.shape, sr_down, sc).astype(self.precision)       
        elif self.sink_mode == "mid":
            sc_down = np.array([data.shape[1] // 2])
            sr_down = np.ones_like(sc_down) * (data.shape[0] - 1)
            seed = self.sub2ind(data.shape, sr_down, sc_down).astype(self.precision)
        elif self.sink_mode == "min":
            # Find the minimum value in the last row
            min_val = np.min(data[-1, :])
            min_idxs = np.where(data[-1, :] == min_val)[0]
            sc_down = min_idxs
            sr_down = np.ones_like(sc_down) * (data.shape[0] - 1)
            seed = self.sub2ind(data.shape, sr_down, sc_down).astype(self.precision)
        elif self.sink_mode == "mask":
            coords = np.where(self.sink_mask != 0)
            sr_down = coords[0]
            sc_down = coords[1]
            seed = self.sub2ind(data.shape, sr_down, sc_down).astype(self.precision)

        seed = np.unique(seed)
        seeds = np.concatenate((seeds, seed))

        # Label 2
        label = np.ones_like(seed) * 2
        labels = np.concatenate((labels, label))

        # Attenuation with Beer-Lambert
        W = self.attenuation_weighting(data, self.alpha)

        # Apply weighting directly to image
        # Same as applying it individually during the formation of the
        # Laplacian
        data = data * W

        # Find condidence values
        map_ = self.confidence_estimation(data, seeds, labels, self.beta, self.gamma)

        if downsample is not None:
            map_ = cv2.resize(map_, (org_W, org_H), interpolation=cv2.INTER_CUBIC)

        return map_

