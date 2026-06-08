import numpy as np
import torch
from plyfile import PlyData

from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.utils.logger import Logger

logger = Logger("PLY_HANDLER")


class PLYHandler:
    def __init__(
        self,
        file_path: str,
    ):
        ply_data = PlyData.read(file_path)

        self.data = ply_data["vertex"]
        self.properties_names = [
            prop.name
            for prop in self.data.properties
        ]  # fmt:skip

    def log_info(self) -> None:
        logger.info(
            "PLY file info:\n"
            f"  Properties: {', '.join(self.properties_names)}\n"
            f"  Number of gaussians: {len(self.data)}",
        )  # fmt:skip

    def get_gaussians(self) -> GaussianCollection:
        means = self._get_means()
        n = len(means)
        # Identity quaternion [w=1, x=0, y=0, z=0] with small isotropic scale
        quaternions = torch.zeros((n, 4), dtype=torch.float32)
        quaternions[:, 0] = 1.0
        scales = torch.full((n, 3), fill_value=1e-3, dtype=torch.float32)
        return GaussianCollection.from_tensors(
            means=means,
            quaternions=quaternions,
            scales=scales,
            colors=self._get_colors(),
            opacities=self._get_opacities(),
        )

    def _get_means(self) -> torch.Tensor:
        positions = np.stack([
            self.data["x"],
            self.data["y"],
            self.data["z"],
        ], axis=1)  # fmt:skip

        return torch.from_numpy(positions).float()

    def _get_colors(self) -> torch.Tensor:
        colors_rgb = np.stack(
            [
                self.data["red"],
                self.data["green"],
                self.data["blue"],
            ],
            axis=1
        ).astype(np.float32) / 255.0  # fmt:skip

        return torch.from_numpy(colors_rgb).float()

    def _get_opacities(self) -> torch.Tensor:
        if "opacity" not in self.properties_names:
            return torch.ones((len(self.data), 1), dtype=torch.float32)

        opacities = self.data["opacity"].astype(np.float32)
        return torch.from_numpy(opacities).float().unsqueeze(1)
