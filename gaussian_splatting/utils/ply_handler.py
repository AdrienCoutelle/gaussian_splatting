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
        return GaussianCollection.from_tensors(
            means=self._get_means(),
            quaternions=self._get_quaternions(),
            scales=self._get_scales(),
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

    def _get_quaternions(self) -> torch.Tensor:
        if (
            "rot_0" not in self.properties_names
            or "rot_1" not in self.properties_names
            or "rot_2" not in self.properties_names
            or "rot_3" not in self.properties_names
        ):  # fmt:skip
            n = len(self.data)
            quaternions = torch.zeros((n, 4), dtype=torch.float32)
            quaternions[:, 0] = 1.0
            return quaternions

        quaternions_np = np.stack([
            self.data["rot_0"],
            self.data["rot_1"],
            self.data["rot_2"],
            self.data["rot_3"],
        ], axis=1)  # fmt:skip
        return torch.from_numpy(quaternions_np).float()

    def _get_scales(self) -> torch.Tensor:
        if (
            "scale_0" not in self.properties_names
            or "scale_1" not in self.properties_names
            or "scale_2" not in self.properties_names
        ):  # fmt:skip
            default_scale = 1e-3
            n = len(self.data)
            return torch.full(
                (n, 3),
                fill_value=default_scale,
                dtype=torch.float32,
            )

        scales_np = np.stack([
            self.data["scale_0"],
            self.data["scale_1"],
            self.data["scale_2"],
        ], axis=1)  # fmt:skip
        return torch.from_numpy(scales_np).float()

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
