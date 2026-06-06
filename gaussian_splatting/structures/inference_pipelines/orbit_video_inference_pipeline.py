import datetime
import os
from typing import Annotated, Literal

import cv2
import numpy as np
import torch
from pydantic import ConfigDict, Field
from tqdm import tqdm

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import Gaussian
from gaussian_splatting.structures.inference_pipelines.base_pipeline import (
    BaseInferencePipeline,
    InferencePipelineParams,
)
from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer


class OrbitPipelineInferenceParams(InferencePipelineParams):
    name: Literal["orbit"]
    model_config = ConfigDict(extra="forbid")

    center: tuple[float, float, float]

    fps: Annotated[int, Field(gt=0)]
    n_frames: Annotated[int, Field(gt=0)]

    min_radius: Annotated[float, Field(gt=0)]
    max_radius: Annotated[float, Field(gt=0)]

    radius_freq: Annotated[float, Field(ge=0)]

    n_revolutions: Annotated[float, Field(gt=0)]

    min_phi: float
    max_phi: float

    phi_freq: Annotated[float, Field(ge=0)]


class OrbitVideoInferencePipeline(BaseInferencePipeline):
    def __init__(
        self,
        renderer: BaseRenderer,
        gaussians: list[Gaussian],
        configuration: OrbitPipelineInferenceParams,
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

        self.gaussians = gaussians
        self.renderer = renderer

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        output_name = (
            f"orbit_video_epoch_{self.epoch}.mp4"
            if self.epoch is not None
            else f"orbit_video_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )

        os.makedirs(self.output_folder, exist_ok=True)

        self.video_writer = cv2.VideoWriter(
            os.path.join(self.output_folder, output_name),
            fourcc,
            self.configuration.fps,
            (self.renderer.config.width, self.renderer.config.height),
        )

        self.radius_list = None
        self.theta_list = None
        self.phi_list = None
        self._initialize_lists()

    def _initialize_lists(self) -> None:
        self.radius_list = [
            self.configuration.min_radius
            + (self.configuration.max_radius - self.configuration.min_radius)
            * 0.5
            * (1 + np.sin(2 * np.pi * self.configuration.radius_freq * i / self.configuration.n_frames))
            for i in range(self.configuration.n_frames)
        ]
        self.theta_list = [
            (360.0 * self.configuration.n_revolutions * i) / self.configuration.n_frames
            for i in range(self.configuration.n_frames)
        ]
        self.phi_list = [
            self.configuration.min_phi
            + (self.configuration.max_phi - self.configuration.min_phi)
            * 0.5
            * (1 + np.sin(2 * np.pi * self.configuration.phi_freq * i / self.configuration.n_frames))
            for i in range(self.configuration.n_frames)
        ]

    def run(self) -> None:
        with torch.no_grad():
            for radius, theta, phi in tqdm(
                zip(
                    self.radius_list,
                    self.theta_list,
                    self.phi_list,
                    strict=True,
                ),
                total=len(self.radius_list),
                desc="Generating video...",
            ):
                pose = self._compute_pose(  # noqa: F841
                    radius,
                    theta,
                    phi,
                )

                camera = Camera(
                    pose=pose,
                    focal_length=self.renderer.config.focal_length,
                    width=self.renderer.config.width,
                    height=self.renderer.config.height,
                )

                frame = self.renderer.render(
                    camera=camera,
                    gaussians=self.gaussians,
                )

                frame_bgr = cv2.cvtColor(
                    (frame.array * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
                self.video_writer.write(frame_bgr)

            self.video_writer.release()

    def _compute_pose(
        self,
        radius: float,
        theta: float,
        phi: float,
    ) -> torch.Tensor:
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)

        x = radius * np.cos(theta_rad) * np.cos(phi_rad)
        y = radius * np.sin(theta_rad) * np.cos(phi_rad)
        z = radius * np.sin(phi_rad)

        cam_pos = np.array([x, y, z]) + self.configuration.center

        return torch.from_numpy(
            self._compute_pose_look_at(
                position=cam_pos,
                look_at=self.configuration.center,
                world_up=np.array([0, 0, 1]),
            )
        )
