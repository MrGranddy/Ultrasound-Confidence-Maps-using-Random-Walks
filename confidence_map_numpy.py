from typing import Literal, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.signal import hilbert


class ConfidenceMap:
    """Confidence map computation class for RF ultrasound data"""

    def __init__(
        self,
        precision: Literal["float16", "float32", "float64"] = "float32",
        alpha: float = 2.0,
        beta: float = 90.0,
        gamma: float = 0.05,
        mode: Literal["RF", "B"] = "B",
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

        # The precision to use for all computations
        self.precision = precision
        self.eps = np.finfo(self.precision).eps

    def xexp(self, inp: np.ndarray, alpha: float) -> np.ndarray:
        """Compute xexp(x, a) = exp(-a * x)"""
        return np.exp(-alpha * inp, dtype=self.precision)

    def normalize(self, inp: np.ndarray) -> np.ndarray:
        """Normalize an array to [0, 1]"""
        return np.divide(
            inp - np.min(inp), (np.max(inp) - np.min(inp) + self.eps), dtype=self.precision
        )

    def attenuation_weighting(self, A: np.ndarray, alpha: float) -> np.ndarray:
        """Compute attenuation weighting

        Args:
            A (np.ndarray): Image
            alpha: Attenuation coefficient (see publication)

        Returns:
            W (np.ndarray): Weighting expresing depth-dependent attenuation
        """

        # Create depth vector
        Dw = np.linspace(0, 1, A.shape[0], dtype=self.precision)
        Dw = np.repeat(
            Dw.reshape(-1, 1), A.shape[1], axis=1
        )  # Repeat depth vector for each column

        W = 1.0 - self.xexp(Dw, alpha)

        return W

    def confidence_laplacian(
        self, P: np.ndarray, A: np.ndarray, beta: float, gamma: float
    ) -> csr_matrix:
        """Compute 6-Connected Laplacian for confidence estimation problem

        Args:
            P (np.ndarray): TODO
            A (np.ndarray): TODO
            beta (float): TODO
            gamma (float): TODO

        Returns:
            L (np.ndarray): TODO
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

        for iter_idx, k in enumerate(
            [
                -1,  # Vertical edges
                1,
                m - 1,  # Diagonal edges
                m + 1,
                -m - 1,
                -m + 1,
                m,  # Horizontal edges
                -m,
            ]
        ):

            Q = P[p + k]

            q = np.where(Q > 0)[0]

            ii = P[p[q]] - 1
            i = np.concatenate((i, ii))
            jj = Q[q] - 1
            j = np.concatenate((j, jj))
            W = np.abs(A[p[ii]] - A[p[jj]])  # Intensity derived weight
            s = np.concatenate((s, W))

            if iter_idx == 1:
                vl = s.shape[0]  # Vertical edges length

        # Normalize weights
        s = self.normalize(s)

        # Horizontal penalty
        s[vl:] += gamma

        # Normalize differences
        s = self.normalize(s)

        # Gaussian weighting function
        s = -((np.exp(-beta * s, dtype=self.precision)) + 1.e-6) # --> This epsilon changes results drastically default: 1.e-6

        # Create Laplacian, diagonal missing
        L = csr_matrix((s, (i, j)))
        print(L.dtype)

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
            A (np.ndarray): Processed image
            seeds (np.ndarray): Seeds for the random walks framework
            labels (np.ndarray): Labels for the random walks framework
            beta: Random walks parameter
            gamma: Horizontal penalty factor

        Returns:
            map: confidence map
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
        N = np.sum(G > 0)
        i_U = np.arange(N)
        i_U[seeds.astype(int)] = 0
        i_U = np.where(i_U > 0)[0]  # Index of unmarked nodes
        B = B[i_U, :]

        # Remove marked nodes from Laplacian by deleting rows and cols
        keep_indices = np.setdiff1d(np.arange(D.shape[0]), seeds)
        D = D[keep_indices, :][:, keep_indices]

        # Adjust labels
        label_adjust = np.min(labels, axis=0, keepdims=True)
        labels = labels - label_adjust + 1  # labels > 0

        # Find number of labels (K)
        labels_present = np.unique(labels)
        number_labels = labels_present.shape[0]

        # Define M matrix
        M = np.zeros((seeds.shape[0], number_labels), dtype=self.precision)
        for k in range(number_labels):
            M[:, k] = labels == labels_present[k]

        # Right-handside (-B^T*M)
        rhs = -B @ M  # type: ignore

        # Solve system
        if number_labels == 2:
            x = spsolve(D, rhs[:, 0])
            x = np.vstack((x, 1.0 - x)).T
        else:
            x = spsolve(D, rhs)

        # Prepare output
        probabilities = np.zeros(
            (N, number_labels), dtype=self.precision
        )  # type: ignore
        for k in range(number_labels):
            # Probabilities for unmarked nodes
            probabilities[i_U, k] = x[:, k]
            # Max probability for marked node of each label
            probabilities[seeds[labels == k].astype(int), k] = 1.0

        # Final reshape with same size as input image (no padding)
        probabilities = probabilities.reshape(
            (A.shape[1], A.shape[0], number_labels)
        ).transpose((1, 0, 2))

        # reshape((A.shape[0], A.shape[1], number_labels))

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

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Compute the confidence map

        Args:
            data (np.ndarray): RF ultrasound data (one scanline per column)

        Returns:
            map (np.ndarray): Confidence map
        """

        print("Preparing confidence estimation...")

        # Normalize data
        data = data.astype(self.precision)
        data = self.normalize(data)

        if self.mode == "RF":
            # MATLAB hilbert applies the Hilbert transform to columns
            data = np.abs(hilbert(data, axis=0)).astype(self.precision)  # type: ignore

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
        sr_down = np.ones_like(sc) * (data.shape[0] - 1)
        seed = self.sub2ind(data.shape, sr_down, sc).astype(self.precision)
        seed = np.unique(seed)
        seeds = np.concatenate((seeds, seed))

        # Label 2
        label = np.ones_like(seed) * 2
        labels = np.concatenate((labels, label))

        # Attenuation with Beer-Lambert
        W = self.attenuation_weighting(data, self.alpha)

        print("Solving confidence estimation problem, please wait...")

        # Apply weighting directly to image
        # Same as applying it individually during the formation of the
        # Laplacian
        data = data * W

        # Find condidence values
        map_ = self.confidence_estimation(data, seeds, labels, self.beta, self.gamma)

        # Only keep probabilities for virtual source notes.
        map_ = map_[:, :, 0]

        return map_