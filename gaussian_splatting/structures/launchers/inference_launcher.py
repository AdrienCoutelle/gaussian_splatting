import numpy as np
import torch

from gaussian_splatting.structures.device import Device
from gaussian_splatting.structures.gaussian import Gaussian
from gaussian_splatting.structures.renderer import GSRenderer


class InferenceLauncher:
    def __init__(
        self,
        file_path: str,
    ):
        self.device = Device.get()
        self.renderer = GSRenderer(
            config=None,
            device=self.device,
        )

        self.gaussians = [
            Gaussian(
                mean=torch.Tensor([0, 0, 0], device=self.device),
                covariance=torch.from_numpy(np.eye(3)),
                color=torch.Tensor((0, 0, 0), device=self.device),
                opacity=torch.Tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.Tensor([0, 0, 1], device=self.device),
                covariance=torch.from_numpy(np.eye(3)),
                color=torch.Tensor((0, 0, 255), device=self.device),
                opacity=torch.Tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.Tensor([0, 1, 0], device=self.device),
                covariance=torch.from_numpy(np.eye(3)),
                color=torch.Tensor((0, 255, 0), device=self.device),
                opacity=torch.Tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.Tensor([0, 1, 1], device=self.device),
                covariance=torch.from_numpy(np.eye(3)),
                color=torch.Tensor((0, 255, 255), device=self.device),
                opacity=torch.Tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.Tensor([1, 0, 0], device=self.device),
                covariance=torch.from_numpy(np.eye(3)),
                color=torch.Tensor((255, 0, 0), device=self.device),
                opacity=torch.Tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.Tensor([1, 0, 1], device=self.device),
                covariance=torch.from_numpy(np.eye(3)),
                color=torch.Tensor((255, 0, 255), device=self.device),
                opacity=torch.Tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.Tensor([1, 1, 0], device=self.device),
                covariance=torch.from_numpy(np.eye(3)),
                color=torch.Tensor((255, 255, 0), device=self.device),
                opacity=torch.Tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.Tensor([1, 1, 1], device=self.device),
                covariance=torch.from_numpy(np.eye(3)),
                color=torch.Tensor((255, 255, 255), device=self.device),
                opacity=torch.Tensor(1, device=self.device),
            ),
        ]

    def run(self) -> None:
        pass
