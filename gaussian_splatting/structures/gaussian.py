from dataclasses import dataclass

import mlx.core as mx


@dataclass
class Gaussian:
    position: mx.array
    quaternion: mx.array  # (4,) as [w, x, y, z]
    scale: mx.array  # (3,)
    sh_coeffs: mx.array  # (num_sh_coeffs, 3) where num_sh_coeffs = (sh_degree + 1)^2
    opacity: mx.array


class GaussianCollection:
    def __init__(
        self,
        gaussians: list[Gaussian],
    ) -> None:
        self.positions = mx.stack([g.position for g in gaussians])
        self.quaternions = mx.stack([g.quaternion for g in gaussians])
        self.scales = mx.stack([g.scale for g in gaussians])
        self.sh_coeffs = mx.stack([g.sh_coeffs for g in gaussians])
        self.opacities = mx.stack([g.opacity for g in gaussians])

    @classmethod
    def from_tensors(
        cls,
        positions: mx.array,
        quaternions: mx.array,
        scales: mx.array,
        sh_coeffs: mx.array,
        opacities: mx.array,
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

    def __getitem__(self, idx):
        return GaussianCollection.from_tensors(
            positions=self.positions[idx],
            quaternions=self.quaternions[idx],
            scales=self.scales[idx],
            sh_coeffs=self.sh_coeffs[idx],
            opacities=self.opacities[idx],
        )
