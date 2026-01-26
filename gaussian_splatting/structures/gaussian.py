import torch


class Gaussian:
    def __init__(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
    ):
        """
        Initialize 3D gaussian.

        Args:
            TODO
        """
        self.mean = mean
        self.covariance = covariance
        self.color = color
        self.opacity = opacity

        assert self.mean.shape == (3,)
        assert self.covariance.shape == (3, 3)
        assert self.color.shape == (3,)
