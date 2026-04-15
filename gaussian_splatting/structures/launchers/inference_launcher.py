import numpy as np
import torch

from gaussian_splatting.structures.device import Device
from gaussian_splatting.structures.gaussian import Gaussian
from gaussian_splatting.structures.inference_pipelines.orbit_video_inference_pipeline import (
    OrbitPipelineInferenceParams,
    OrbitVideoInferencePipeline,
)
from gaussian_splatting.structures.renderer import GSRenderer, GSRendererConfig
from gaussian_splatting.utils.logger import Logger

logger = Logger("INFERENCE_LAUNCHER")


class InferenceLauncher:
    def __init__(
        self,
        file_path: str,
    ):
        self.device = Device.get()
        self.renderer = GSRenderer(
            config=GSRendererConfig(),
            device=self.device,
        )

    def run(self) -> None:
        pipeline_config = OrbitPipelineInferenceParams(
            center=[0, 0, 0],
            fps=30,
            n_frames=150,
            min_radius=2.0,
            max_radius=3.0,
            radius_freq=0.5,
            n_revolutions=2,
            min_phi=0,
            max_phi=180,
            phi_freq=0.5,
        )

        pipeline = OrbitVideoInferencePipeline(
            gaussians=self._create_gaussians(),
            configuration=pipeline_config,
            device=self.device,
            output_folder="./output",
        )

        pipeline.run()

    def _create_gaussians(
        self,
        cube_size: float = 2.0,
        points_per_side: int = 10,
    ) -> list[Gaussian]:
        """Create gaussians on a cube surface with different colors for each face."""
        gaussians = []

        # Shared covariance for all gaussians
        covariance = torch.from_numpy(np.eye(3, dtype=np.float32)).to(self.device) / 100

        # Define colors for each face: front, back, left, right, top, bottom
        face_colors = {
            "front": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),  # Red
            "back": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device),  # Green
            "left": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device),  # Blue
            "right": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32, device=self.device),  # Yellow
            "top": torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32, device=self.device),  # Magenta
            "bottom": torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32, device=self.device),  # Cyan
        }

        # Generate grid points on each face
        half = cube_size / 2
        grid = np.linspace(-half, half, points_per_side, dtype=np.float32)

        # Front face (z = half)
        for x in grid:
            for y in grid:
                gaussians.append(
                    Gaussian(
                        mean=torch.tensor([x, y, half], dtype=torch.float32, device=self.device),
                        covariance=covariance,
                        color=face_colors["front"],
                        opacity=torch.tensor(1.0, dtype=torch.float32, device=self.device),
                    )
                )  # fmt:skip

        # Back face (z = -half)
        for x in grid:
            for y in grid:
                gaussians.append(
                    Gaussian(
                        mean=torch.tensor([x, y, -half], dtype=torch.float32, device=self.device),
                        covariance=covariance,
                        color=face_colors["back"],
                        opacity=torch.tensor(1.0, dtype=torch.float32, device=self.device),
                    )
                )  # fmt:skip

        # Left face (x = -half)
        for z in grid:
            for y in grid:
                gaussians.append(
                    Gaussian(
                        mean=torch.tensor([-half, y, z], dtype=torch.float32, device=self.device),
                        covariance=covariance,
                        color=face_colors["left"],
                        opacity=torch.tensor(1.0, dtype=torch.float32, device=self.device),
                    )
                )  # fmt:skip

        # Right face (x = half)
        for z in grid:
            for y in grid:
                gaussians.append(
                    Gaussian(
                        mean=torch.tensor([half, y, z], dtype=torch.float32, device=self.device),
                        covariance=covariance,
                        color=face_colors["right"],
                        opacity=torch.tensor(1.0, dtype=torch.float32, device=self.device),
                    )
                )  # fmt:skip

        # Top face (y = half)
        for x in grid:
            for z in grid:
                gaussians.append(
                    Gaussian(
                        mean=torch.tensor([x, half, z], dtype=torch.float32, device=self.device),
                        covariance=covariance,
                        color=face_colors["top"],
                        opacity=torch.tensor(1.0, dtype=torch.float32, device=self.device),
                    )
                )  # fmt:skip

        # Bottom face (y = -half)
        for x in grid:
            for z in grid:
                gaussians.append(
                    Gaussian(
                        mean=torch.tensor([x, -half, z], dtype=torch.float32, device=self.device),
                        covariance=covariance,
                        color=face_colors["bottom"],
                        opacity=torch.tensor(1.0, dtype=torch.float32, device=self.device),
                    )
                )  # fmt:skip

        logger.info(f"Created {len(gaussians)} gaussians on the cube surface.")

        return gaussians
