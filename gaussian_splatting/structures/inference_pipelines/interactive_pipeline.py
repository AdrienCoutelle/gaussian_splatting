import datetime
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import Gaussian
from gaussian_splatting.structures.inference_pipelines.base_pipeline import (
    BaseInferencePipeline,
    InferencePipelineParams,
)
from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer


@dataclass
class InteractiveInferencePipelineParams(InferencePipelineParams):
    initial_look_at: list[float]
    initial_position: list[float]

    @staticmethod
    def from_dict(configuration: dict) -> "InteractiveInferencePipelineParams":
        if not isinstance(configuration, dict):
            raise ValueError(
                f"InteractiveInferencePipelineParams must be a dictionary, got '{type(configuration).__name__}'."
            )

        mandatory_fields = {
            "initial_look_at",
            "initial_position",
        }

        if not set(configuration.keys()).issuperset(mandatory_fields):
            missing_fields = mandatory_fields - set(configuration.keys())
            raise ValueError(
                f"InteractiveInferencePipelineParams is missing the following mandatory fields: {missing_fields}, "
                f"got {set(configuration.keys())}."
            )

        look_at = configuration["initial_look_at"]
        if not isinstance(look_at, list) or len(look_at) != 3 or not all(isinstance(c, (int, float)) for c in look_at):
            raise ValueError(
                f"InteractiveInferencePipelineParams 'initial_look_at' "
                f"must be a list of three numbers, got '{look_at}'."
            )

        position = configuration["initial_position"]
        if (
            not isinstance(position, list)
            or len(position) != 3
            or not all(isinstance(c, (int, float)) for c in position)
        ):
            raise ValueError(
                f"InteractiveInferencePipelineParams 'initial_position' "
                f"must be a list of three numbers, got '{position}'."
            )

        return InteractiveInferencePipelineParams(
            initial_look_at=look_at,
            initial_position=position,
        )


class InteractiveInferencePipeline(BaseInferencePipeline):
    def __init__(
        self,
        renderer: BaseRenderer,
        gaussians: list[Gaussian],
        configuration: InteractiveInferencePipelineParams,
        device: torch.device,
        output_folder: str,
        epoch: int | None = None,
    ):
        super().__init__(
            configuration=configuration,
            device=device,
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

        self.device = device

    def run(self) -> None:
        with torch.no_grad():
            pose = torch.from_numpy(  # noqa: F841
                self._compute_pose_look_at(
                    position=np.array(self.configuration.position),
                    look_at=np.array(self.configuration.look_at),
                    world_up=np.array([0, 0, 1]),
                )
            )

            camera = Camera(
                pose=pose,
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
