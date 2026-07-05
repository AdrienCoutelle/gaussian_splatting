import mlx.core as mx
import numpy as np
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
        self.properties_names = [prop.name for prop in self.data.properties]

    def log_info(self) -> None:
        logger.info(
            "PLY file info:\n"
            f"  Properties: {', '.join(self.properties_names)}\n"
            f"  Number of gaussians: {len(self.data)}",
        )

    def get_gaussians(self) -> GaussianCollection:
        n = len(self.data)

        positions = np.stack(
            [self.data["x"], self.data["y"], self.data["z"]],
            axis=1,
        )

        quaternions = np.stack(
            [
                self.data["rot_0"],
                self.data["rot_1"],
                self.data["rot_2"],
                self.data["rot_3"],
            ],
            axis=1,
        )

        scales = np.stack(
            [
                self.data["scale_0"],
                self.data["scale_1"],
                self.data["scale_2"],
            ],
            axis=1,
        )

        opacities = self.data["opacity"].astype(np.float32).reshape(-1, 1)

        SH_DEGREE = 3
        NUM_SH_COEFFS = (SH_DEGREE + 1) ** 2
        NUM_SH_REST = NUM_SH_COEFFS - 1

        # DC coefficients
        sh_dc = np.stack(
            [
                self.data["f_dc_0"],
                self.data["f_dc_1"],
                self.data["f_dc_2"],
            ],
            axis=1,
        )

        num_rest_in_file = len([name for name in self.properties_names if name.startswith("f_rest_")]) // 3

        sh_rest = np.zeros((n, NUM_SH_REST, 3), dtype=np.float32)

        for c in range(3):
            for k in range(num_rest_in_file):
                sh_rest[:, k, c] = self.data[f"f_rest_{c * num_rest_in_file + k}"]

        sh_dc = mx.expand_dims(mx.array(sh_dc, dtype=mx.float32), axis=1)
        sh_rest = mx.array(sh_rest, dtype=mx.float32)

        sh_coeffs = mx.concatenate([sh_dc, sh_rest], axis=1)

        return GaussianCollection.from_tensors(
            positions=mx.array(positions, dtype=mx.float32),
            quaternions=mx.array(quaternions, dtype=mx.float32),
            scales=mx.array(scales, dtype=mx.float32),
            sh_coeffs=sh_coeffs,
            opacities=mx.array(opacities, dtype=mx.float32),
        )
