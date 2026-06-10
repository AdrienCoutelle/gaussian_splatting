import numpy as np
import torch
from plyfile import PlyData

from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.utils.logger import Logger

logger = Logger("PLY_LOADER")


class PLYLoader:
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
        n = len(self.data)

        positions = np.stack([self.data["x"], self.data["y"], self.data["z"]], axis=1)
        quaternions = np.stack([self.data["rot_0"], self.data["rot_1"], self.data["rot_2"], self.data["rot_3"]], axis=1)
        scales = np.stack([self.data["scale_0"], self.data["scale_1"], self.data["scale_2"]], axis=1)
        opacities = self.data["opacity"].astype(np.float32).reshape(-1, 1)

        sh_dc = np.stack([self.data["f_dc_0"], self.data["f_dc_1"], self.data["f_dc_2"]], axis=1)  # (N, 3)
        num_rest = len([name for name in self.properties_names if name.startswith("f_rest_")]) // 3
        sh_dc_tensor = torch.from_numpy(sh_dc.astype(np.float32)).unsqueeze(1)  # (N, 1, 3)
        if num_rest > 0:
            sh_rest = np.zeros((n, num_rest, 3), dtype=np.float32)
            for c in range(3):
                for k in range(num_rest):
                    sh_rest[:, k, c] = self.data[f"f_rest_{c * num_rest + k}"]
            sh_coeffs = torch.cat([sh_dc_tensor, torch.from_numpy(sh_rest)], dim=1)  # (N, num_sh_coeffs, 3)
        else:
            sh_coeffs = sh_dc_tensor

        return GaussianCollection.from_tensors(
            positions=torch.from_numpy(positions.astype(np.float32)),
            quaternions=torch.from_numpy(quaternions.astype(np.float32)),
            scales=torch.from_numpy(scales.astype(np.float32)),
            sh_coeffs=sh_coeffs,
            opacities=torch.from_numpy(opacities),
        )
