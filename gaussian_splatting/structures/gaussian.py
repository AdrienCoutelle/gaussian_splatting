from dataclasses import dataclass

import torch


@dataclass
class Gaussian:
    mean: torch.Tensor
    covariance: torch.Tensor
    color: torch.Tensor
    opacity: torch.Tensor


class GaussianCollection:
    def __init__(
        self,
        gaussians: list[Gaussian],
    ) -> None:
        self.means = torch.stack([g.mean for g in gaussians])
        self.covariances = torch.stack([g.covariance for g in gaussians])
        self.colors = torch.stack([g.color for g in gaussians])
        self.opacities = torch.stack([g.opacity for g in gaussians])

    @classmethod
    def from_tensors(
        cls,
        means: torch.Tensor,
        covariances: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
    ) -> "GaussianCollection":
        collection = cls.__new__(cls)
        collection.means = means
        collection.covariances = covariances
        collection.colors = colors
        collection.opacities = opacities

        return collection

    @property
    def opacites(self) -> torch.Tensor:
        return self.opacities

    def __len__(self) -> int:
        return int(self.means.shape[0])
