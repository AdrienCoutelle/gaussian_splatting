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
    ):
        self.means = torch.stack([g.mean for g in gaussians])
        self.covariances = torch.stack([g.covariance for g in gaussians])
        self.colors = torch.stack([g.color for g in gaussians])
        self.opacites = torch.stack([g.opacity for g in gaussians])
