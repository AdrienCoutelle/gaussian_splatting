import numpy as np
import torch

from gaussian_splatting.structures.device import Device
from gaussian_splatting.structures.gaussian import Gaussian
from gaussian_splatting.structures.inference_pipelines.orbit_video_inference_pipeline import (
    OrbitPipelineInferenceParams,
    OrbitVideoInferencePipeline,
)
from gaussian_splatting.structures.renderer import GSRenderer, GSRendererConfig


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

        self.gaussians = [
            Gaussian(
                mean=torch.tensor([0, 0, 0], device=self.device),
                covariance=torch.from_numpy(np.eye(3)) / 100,
                color=torch.tensor((0, 0, 0), device=self.device),
                opacity=torch.tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.tensor([0, 0, 1], device=self.device),
                covariance=torch.from_numpy(np.eye(3)) / 100,
                color=torch.tensor((0, 0, 255), device=self.device),
                opacity=torch.tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.tensor([0, 1, 0], device=self.device),
                covariance=torch.from_numpy(np.eye(3)) / 100,
                color=torch.tensor((0, 255, 0), device=self.device),
                opacity=torch.tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.tensor([0, 1, 1], device=self.device),
                covariance=torch.from_numpy(np.eye(3)) / 100,
                color=torch.tensor((0, 255, 255), device=self.device),
                opacity=torch.tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.tensor([1, 0, 0], device=self.device),
                covariance=torch.from_numpy(np.eye(3)) / 100,
                color=torch.tensor((255, 0, 0), device=self.device),
                opacity=torch.tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.tensor([1, 0, 1], device=self.device),
                covariance=torch.from_numpy(np.eye(3)) / 100,
                color=torch.tensor((255, 0, 255), device=self.device),
                opacity=torch.tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.tensor([1, 1, 0], device=self.device),
                covariance=torch.from_numpy(np.eye(3)) / 100,
                color=torch.tensor((255, 255, 0), device=self.device),
                opacity=torch.tensor(1, device=self.device),
            ),
            Gaussian(
                mean=torch.tensor([1, 1, 1], device=self.device),
                covariance=torch.from_numpy(np.eye(3)) / 100,
                color=torch.tensor((255, 255, 255), device=self.device),
                opacity=torch.tensor(1, device=self.device),
            ),
        ]

    def run(self) -> None:
        pipeline_config = OrbitPipelineInferenceParams(
            center=[0, 0, 0],
            fps=30,
            n_frames=300,
            min_radius=1.0,
            max_radius=2.0,
            radius_freq=0.5,
            n_revolutions=2,
            min_phi=0,
            max_phi=180,
            phi_freq=0.5,
        )

        pipeline = OrbitVideoInferencePipeline(
            gaussians=self.gaussians,
            configuration=pipeline_config,
            device=self.device,
            output_folder="./output",
        )

        pipeline.run()
