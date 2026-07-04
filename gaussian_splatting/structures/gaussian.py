from dataclasses import dataclass

import torch


@dataclass
class Gaussian:
    position: torch.Tensor
    quaternion: torch.Tensor  # (4,) as [w, x, y, z]
    scale: torch.Tensor  # (3,)
    sh_coeffs: torch.Tensor  # (num_sh_coeffs, 3) where num_sh_coeffs = (sh_degree + 1)^2
    opacity: torch.Tensor


class GaussianCollection:
    def __init__(
        self,
        gaussians: list[Gaussian],
    ) -> None:
        self.positions = torch.stack([g.position for g in gaussians])
        self.quaternions = torch.stack([g.quaternion for g in gaussians])
        self.scales = torch.stack([g.scale for g in gaussians])
        self.sh_coeffs = torch.stack([g.sh_coeffs for g in gaussians])
        self.opacities = torch.stack([g.opacity for g in gaussians])

    @classmethod
    def from_tensors(
        cls,
        positions: torch.Tensor,
        quaternions: torch.Tensor,
        scales: torch.Tensor,
        sh_coeffs: torch.Tensor,
        opacities: torch.Tensor,
    ) -> "GaussianCollection":
        collection = cls.__new__(cls)
        collection.positions = positions
        collection.quaternions = quaternions
        collection.scales = scales
        collection.sh_coeffs = sh_coeffs
        collection.opacities = opacities

        return collection

    def to_list(self) -> list[Gaussian]:
        """Convert the collection back to a list of individual Gaussians."""
        return [
            Gaussian(
                position=self.positions[i],
                quaternion=self.quaternions[i],
                scale=self.scales[i],
                sh_coeffs=self.sh_coeffs[i],
                opacity=self.opacities[i],
            )
            for i in range(len(self))
        ]  # fmt:skip

    def __len__(self) -> int:
        return int(self.positions.shape[0])
