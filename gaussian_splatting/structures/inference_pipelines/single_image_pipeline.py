import datetime
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from gaussian_splatting.structures.inference_pipelines.base_pipeline import (
    BaseInferencePipeline,
    InferencePipelineParams,
)


@dataclass
class SingleImageInferencePipelineParams(InferencePipelineParams):
    look_at: list[float]
    position: list[float]

    @staticmethod
    def from_dict(configuration: dict) -> "SingleImageInferencePipelineParams":
        if not isinstance(configuration, dict):
            raise ValueError(
                f"SingleImageInferencePipelineParams must be a dictionary, got '{type(configuration).__name__}'."
            )

        mandatory_fields = {
            "look_at",
            "position",
        }

        if not set(configuration.keys()).issuperset(mandatory_fields):
            missing_fields = mandatory_fields - set(configuration.keys())
            raise ValueError(
                f"SingleImageInferencePipelineParams is missing the following mandatory fields: {missing_fields}, "
                f"got {set(configuration.keys())}."
            )

        look_at = configuration["look_at"]
        if not isinstance(look_at, list) or len(look_at) != 3 or not all(isinstance(c, (int, float)) for c in look_at):
            raise ValueError(
                f"SingleImageInferencePipelineParams 'look_at' must be a list of three numbers, got '{look_at}'."
            )

        position = configuration["position"]
        if (
            not isinstance(position, list)
            or len(position) != 3
            or not all(isinstance(c, (int, float)) for c in position)
        ):
            raise ValueError(
                f"SingleImageInferencePipelineParams 'position' must be a list of three numbers, got '{position}'."
            )

        return SingleImageInferencePipelineParams(
            look_at=look_at,
            position=position,
        )


class SingleImageInferencePipeline(BaseInferencePipeline):
    def __init__(
        self,
        configuration: SingleImageInferencePipelineParams,
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

            rendered_image = None  # TODO

            cv2.imwrite(
                filename=self.output_path,
                img=rendered_image,
            )
