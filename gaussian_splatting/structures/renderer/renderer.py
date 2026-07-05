from dataclasses import dataclass

import mlx
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import Gaussian, GaussianCollection


@dataclass
class Image:
    array: np.ndarray

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]


class RendererConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int
    height: int
    focal_length: float


class Renderer:
    def __init__(
        self,
        config: RendererConfig,
    ) -> None:
        self.config = config

    def render(
        self,
        camera: Camera,
        gaussians: list[Gaussian] | GaussianCollection,
    ) -> Image:
        return Image(
            array=self.render_tensor(
                camera=camera,
                gaussians=gaussians,
            )
            .detach()
            .cpu()
            .numpy(),
        )

    def render_tensor(
        self,
        camera: Camera,
        gaussians: list[Gaussian] | GaussianCollection,
    ) -> torch.Tensor:
        """Render to a float tensor (H, W, 3) in [0, 1], keeping the computation graph intact."""
        if isinstance(gaussians, GaussianCollection):
            gaussian_collection = gaussians
        else:
            gaussian_collection = GaussianCollection(gaussians=gaussians)

        camera_space_gaussians = self._transform_to_camera_space(
            camera=camera,
            gaussians=gaussian_collection,
        )

        screen_space_gaussians = self._project_to_screen_space(
            camera=camera,
            gaussians=camera_space_gaussians,
        )

        output_image = torch.zeros(
            (camera.h, camera.w, 3),
            dtype=torch.float32,
        )

        if screen_space_gaussians is None:
            return output_image

        sorted_indices = torch.argsort(screen_space_gaussians.depths, descending=False)

        self._splat_gaussians_vectorized(
            image=output_image,
            gaussians=screen_space_gaussians,
            sorted_indices=sorted_indices,
            image_height=camera.h,
            image_width=camera.w,
        )

        return output_image.clamp(0.0, 1.0)
