from gaussian_splatting.structures.device import Device
from gaussian_splatting.structures.inference_pipelines.orbit_video_inference_pipeline import (
    OrbitPipelineInferenceParams,
    OrbitVideoInferencePipeline,
)
from gaussian_splatting.structures.renderer import GSRenderer, GSRendererConfig
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply_loader import load_ply_gaussians
from gaussian_splatting.utils.profiler import Profiler

logger = Logger("INFERENCE_LAUNCHER")


class InferenceLauncher:
    def __init__(
        self,
        ply_path: str,
    ):
        self.ply_path = ply_path
        self.device = Device.get()
        self.renderer = GSRenderer(
            config=GSRendererConfig(),
            device=self.device,
        )

    def run(self) -> None:
        logger.info(f"Loading Gaussians from {self.ply_path}...")
        gaussian_collection = load_ply_gaussians(ply_path=self.ply_path)
        logger.info(f"Loaded {len(gaussian_collection)} Gaussians")

        gaussians = gaussian_collection.to_list()

        pipeline_config = OrbitPipelineInferenceParams(
            center=[0.0378, -0.0912, -0.0017],
            fps=30,
            n_frames=1,
            min_radius=0.3,
            max_radius=0.5,
            radius_freq=0.5,
            n_revolutions=2,
            min_phi=0,
            max_phi=180,
            phi_freq=0.5,
        )

        pipeline = OrbitVideoInferencePipeline(
            gaussians=gaussians,
            configuration=pipeline_config,
            device=self.device,
            output_folder="./output",
        )

        pipeline.run()

        Profiler.print_stats()
