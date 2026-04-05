import datetime
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from tqdm import tqdm

from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.inference_pipelines.base_pipeline import (
    BaseInferencePipeline,
    InferencePipelineParams,
)
from gaussian_splatting.structures.renderer import Camera, GSRenderer, GSRendererConfig


@dataclass
class OrbitPipelineInferenceParams(InferencePipelineParams):
    center: list[float]

    fps: int
    n_frames: int
    min_radius: float
    max_radius: float
    radius_freq: float
    n_revolutions: int | float
    min_phi: float
    max_phi: float
    phi_freq: float

    @staticmethod
    def from_dict(configuration: dict) -> "OrbitPipelineInferenceParams":
        if not isinstance(configuration, dict):
            raise ValueError(
                f"OrbitPipelineInferenceParams must be a dictionary, got '{type(configuration).__name__}'."
            )

        mandatory_fields = {
            "center",
            "fps",
            "n_frames",
            "min_radius",
            "max_radius",
            "radius_freq",
            "n_revolutions",
            "min_phi",
            "max_phi",
            "phi_freq",
        }

        if not set(configuration.keys()).issuperset(mandatory_fields):
            missing_fields = mandatory_fields - set(configuration.keys())
            raise ValueError(
                f"OrbitPipelineInferenceParams is missing the following mandatory fields: {', '.join(missing_fields)}, "
                f"got {', '.join(configuration.keys())}."
            )

        center = configuration["center"]
        if not isinstance(center, list) or len(center) != 3 or not all(isinstance(c, (int, float)) for c in center):
            raise ValueError(f"OrbitPipelineInferenceParams 'center' must be a list of three numbers, got '{center}'.")

        fps = configuration["fps"]
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError(f"OrbitPipelineInferenceParams 'fps' must be a positive integer, got '{fps}'.")

        n_frames = configuration["n_frames"]
        if not isinstance(n_frames, int) or n_frames <= 0:
            raise ValueError(f"OrbitPipelineInferenceParams 'n_frames' must be a positive integer, got '{n_frames}'.")

        min_radius = configuration["min_radius"]
        if not isinstance(min_radius, (int, float)) or min_radius <= 0:
            raise ValueError(
                f"OrbitPipelineInferenceParams 'min_radius' must be a positive number, got '{min_radius}'."
            )

        max_radius = configuration["max_radius"]
        if not isinstance(max_radius, (int, float)) or max_radius <= 0:
            raise ValueError(
                f"OrbitPipelineInferenceParams 'max_radius' must be a positive number, got '{max_radius}'."
            )

        radius_freq = configuration["radius_freq"]
        if not isinstance(radius_freq, (int, float)) or radius_freq < 0:
            raise ValueError(
                f"OrbitPipelineInferenceParams 'radius_freq' must be a non-negative number, got '{radius_freq}'."
            )

        n_revolutions = configuration["n_revolutions"]
        if not isinstance(n_revolutions, (int, float)) or n_revolutions <= 0:
            raise ValueError(
                f"OrbitPipelineInferenceParams 'n_revolutions' must be a positive number, got '{n_revolutions}'."
            )

        min_phi = configuration["min_phi"]
        if not isinstance(min_phi, (int, float)):
            raise ValueError(f"OrbitPipelineInferenceParams 'min_phi' must be a number, got '{min_phi}'.")

        max_phi = configuration["max_phi"]
        if not isinstance(max_phi, (int, float)):
            raise ValueError(f"OrbitPipelineInferenceParams 'max_phi' must be a number, got '{max_phi}'.")

        phi_freq = configuration["phi_freq"]
        if not isinstance(phi_freq, (int, float)) or phi_freq < 0:
            raise ValueError(
                f"OrbitPipelineInferenceParams 'phi_freq' must be a non-negative number, got '{phi_freq}'."
            )

        return OrbitPipelineInferenceParams(
            center=center,
            fps=fps,
            n_frames=n_frames,
            min_radius=min_radius,
            max_radius=max_radius,
            radius_freq=radius_freq,
            n_revolutions=n_revolutions,
            min_phi=min_phi,
            max_phi=max_phi,
            phi_freq=phi_freq,
        )


class OrbitVideoInferencePipeline(BaseInferencePipeline):
    def __init__(
        self,
        gaussian_collection: GaussianCollection,
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

        self.gaussian_collection = gaussian_collection

        self.renderer = GSRenderer(
            config=GSRendererConfig(),
            device=device,
        )

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
            (1000, 1000),
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
            for i, (radius, theta, phi) in tqdm(
                enumerate(
                    zip(
                        self.radius_list,
                        self.theta_list,
                        self.phi_list,
                        strict=True,
                    )
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
                    focal_length=100,
                    width=1000,
                    height=1000,
                )

                frame = self.renderer.render(
                    camera=camera,
                    gaussian_collection=self.gaussian_collection,
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
