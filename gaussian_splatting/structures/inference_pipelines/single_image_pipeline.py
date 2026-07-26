import datetime
import os
from typing import Literal

import cv2
import mlx.core as mx
import numpy as np
from pydantic import ConfigDict

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.inference_pipelines.base_pipeline import (
    BaseInferencePipeline,
    InferencePipelineParams,
)
from gaussian_splatting.structures.renderer.renderer import Renderer


class SingleImageInferencePipelineParams(InferencePipelineParams):
    name: Literal["single_image"]
    model_config = ConfigDict(extra="forbid")

    position: tuple[float, float, float]
    look_at: tuple[float, float, float] | Literal["mean"] = "mean"


class SingleImageInferencePipeline(BaseInferencePipeline):
    def __init__(
        self,
        renderer: Renderer,
        gaussians: GaussianCollection,
        configuration: SingleImageInferencePipelineParams,
        output_folder: str,
        epoch: int | None = None,
    ):
        super().__init__(
            configuration=configuration,
            output_folder=output_folder,
            epoch=epoch,
        )

        self.renderer = renderer
        self.gaussians = gaussians

        os.makedirs(self.output_folder, exist_ok=True)

        output_name = (
            f"position_epoch_{self.epoch}.jpg"
            if self.epoch is not None
            else f"single_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )

        self.output_path = os.path.join(
            self.output_folder,
            output_name,
        )

        if self.configuration.look_at == "mean":
            self.configuration.look_at = mx.mean(self.gaussians.positions, axis=0)

    def run(self) -> None:
        pose = self._compute_pose_look_at(
            position=np.array(self.configuration.position),
            look_at=np.array(self.configuration.look_at),
            world_up=np.array([0, 0, 1]),
        )

        camera = Camera(
            pose=mx.array(pose),
            focal_length=self.renderer.config.focal_length,
            width=self.renderer.config.width,
            height=self.renderer.config.height,
        )

        rendered_image = self.renderer.render(
            camera=camera,
            gaussians=self.gaussians,
        )

        image_array = (rendered_image.array * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        cv2.imwrite(
            filename=self.output_path,
            img=image_bgr,
        )
