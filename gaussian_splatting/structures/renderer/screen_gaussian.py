from dataclasses import dataclass

import mlx.core as mx


@dataclass
class ScreenSpaceGaussians:
    means_2d: mx.array
    covariances_2d: mx.array
    depths: mx.array
    colors: mx.array
    opacities: mx.array

    def __len__(self) -> int:
        return int(self.means_2d.shape[0])

    def __getitem__(self, idx) -> "ScreenSpaceGaussians":
        return ScreenSpaceGaussians(
            means_2d=self.means_2d[idx],
            covariances_2d=self.covariances_2d[idx],
            depths=self.depths[idx],
            colors=self.colors[idx],
            opacities=self.opacities[idx],
        )

    def _unpack_covariances(self) -> tuple[mx.array, mx.array, mx.array]:
        """Unpacks 2D covariance matrices into components a, b, and c."""
        if self.covariances_2d.ndim == 3:
            # Assumes shape [N, 2, 2]
            a = self.covariances_2d[:, 0, 0]
            b = self.covariances_2d[:, 0, 1]
            c = self.covariances_2d[:, 1, 1]
        else:
            # Assumes shape [N, 3] corresponding to [a, b, c]
            a = self.covariances_2d[:, 0]
            b = self.covariances_2d[:, 1]
            c = self.covariances_2d[:, 2]
        return a, b, c

    @property
    def conics(self) -> mx.array:
        """
        Computes the inverse of the 2D covariance matrix.
        Returns shape (N, 3) representing [inv_a, inv_b, inv_c].
        """
        a, b, c = self._unpack_covariances()

        det = a * c - b * b
        det = mx.maximum(det, 1e-6)  # Avoid division by zero

        inv_a = c / det
        inv_b = -b / det
        inv_c = a / det

        return mx.stack([inv_a, inv_b, inv_c], axis=-1)

    def compute_max_extent(self, extent_factor: float) -> mx.array:
        """
        Computes the screen-space bounding radius for each Gaussian
        scaled by the provided extent_factor.

        Returns shape (N,).
        """
        a, b, c = self._unpack_covariances()

        # Compute eigenvalues of the 2D covariance matrix
        trace = a + c
        term = mx.sqrt(mx.maximum(((a - c) * 0.5) ** 2 + b**2, 0.0))
        lambda_max = (trace * 0.5) + term

        # Calculate standard deviation scaled by the extent factor
        return extent_factor * mx.sqrt(mx.maximum(lambda_max, 1e-6))
