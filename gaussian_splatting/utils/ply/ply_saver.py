import numpy as np
from plyfile import PlyData, PlyElement

from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.utils.logger import Logger

logger = Logger("PLY_SAVER")


class PLYSaver:
    def __init__(
        self,
        file_path: str,
    ):
        self.file_path = file_path

    def save_gaussians(
        self,
        gaussians_collection: GaussianCollection,
    ) -> None:
        positions = np.array(gaussians_collection.positions, dtype=np.float32)  # (N, 3)
        quaternions = np.array(gaussians_collection.quaternions, dtype=np.float32)  # (N, 4)
        scales = np.array(gaussians_collection.scales, dtype=np.float32)  # (N, 3)
        sh_coeffs = np.array(gaussians_collection.sh_coeffs, dtype=np.float32)  # (N, num_sh, 3)
        opacities = np.array(gaussians_collection.opacities, dtype=np.float32)  # (N, 1)
        if opacities.ndim == 1:
            opacities = opacities[:, np.newaxis]

        n = positions.shape[0]

        sh_dc = sh_coeffs[:, 0, :]  # (N, 3)
        sh_rest = sh_coeffs[:, 1:, :]  # (N, num_sh_coeffs - 1, 3)
        num_rest = sh_rest.shape[1]

        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            *[(f"f_rest_{i}", "f4") for i in range(num_rest * 3)],
            ("opacity", "f4"),
        ]  # fmt:skip

        vertex_data = np.zeros(n, dtype=dtype)

        vertex_data["x"] = positions[:, 0]
        vertex_data["y"] = positions[:, 1]
        vertex_data["z"] = positions[:, 2]
        vertex_data["rot_0"] = quaternions[:, 0]
        vertex_data["rot_1"] = quaternions[:, 1]
        vertex_data["rot_2"] = quaternions[:, 2]
        vertex_data["rot_3"] = quaternions[:, 3]
        vertex_data["scale_0"] = scales[:, 0]
        vertex_data["scale_1"] = scales[:, 1]
        vertex_data["scale_2"] = scales[:, 2]
        vertex_data["f_dc_0"] = sh_dc[:, 0]
        vertex_data["f_dc_1"] = sh_dc[:, 1]
        vertex_data["f_dc_2"] = sh_dc[:, 2]
        # f_rest stored channel-first: all R coefficients, then G, then B
        for c in range(3):
            for k in range(num_rest):
                vertex_data[f"f_rest_{c * num_rest + k}"] = sh_rest[:, k, c]
        vertex_data["opacity"] = opacities[:, 0]

        element = PlyElement.describe(vertex_data, "vertex")
        PlyData([element]).write(self.file_path)

        logger.info(f"Saved {n} gaussians to {self.file_path}")
