from dataclasses import dataclass

from gaussian_splatting.structures.device import Device
from gaussian_splatting.structures.inference_pipelines.factory import InferencePipelineConfig, InferencePipelineFactory
from gaussian_splatting.structures.renderer import GSRenderer, GSRendererConfig
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply_loader import load_ply_gaussians
from gaussian_splatting.utils.profiler import Profiler

logger = Logger("INFERENCE_LAUNCHER")


@dataclass
class InferenceConfig:
    ply_file_path: str

    inference_pipeline_config: InferencePipelineConfig

    output_folder: str

    @staticmethod
    def from_dict(configuration: dict) -> "InferenceConfig":
        if not isinstance(configuration, dict):
            raise ValueError(f"InferenceConfig must be a dictionary, got '{type(configuration).__name__}'.")

        mandatory_fields = {
            "ply_file_path",
            "output_folder",
            "inference_pipeline_config",
        }

        if not set(configuration.keys()).issuperset(mandatory_fields):
            missing_fields = mandatory_fields - set(configuration.keys())
            raise ValueError(
                f"InferenceConfig is missing the following mandatory fields: {', '.join(missing_fields)}, "
                f"got {', '.join(configuration.keys())}."
            )

        ply_file_path = configuration["ply_file_path"]
        if not isinstance(ply_file_path, str):
            raise ValueError(f"InferenceConfig 'ply_file_path' must be a string, got '{ply_file_path}'.")

        output_folder = configuration["output_folder"]
        if not isinstance(output_folder, str):
            raise ValueError(f"InferenceConfig 'output_folder' must be a string, got '{output_folder}'.")

        inference_pipeline_config = InferencePipelineConfig.from_dict(configuration["inference_pipeline_config"])

        return InferenceConfig(
            ply_file_path=ply_file_path,
            inference_pipeline_config=inference_pipeline_config,
            output_folder=output_folder,
        )


class InferenceLauncher:
    def __init__(
        self,
        config: InferenceConfig,
    ):
        self.config = config
        self.device = Device.get()

        self.renderer = GSRenderer(
            config=GSRendererConfig(),
            device=self.device,
        )

        logger.info(f"Loading Gaussians from {self.config.ply_file_path}...")
        gaussian_collection = load_ply_gaussians(ply_path=self.config.ply_file_path)
        logger.info(f"Loaded {len(gaussian_collection)} Gaussians")

        gaussians = gaussian_collection.to_list()

        self.pipeline = InferencePipelineFactory.create(
            gaussians=gaussians,
            configuration=self.config.inference_pipeline_config,
            device=self.device,
            output_folder=self.config.output_folder,
        )

        logger.warning("WOUHOUUU")

    def run(self) -> None:
        logger.info("Starting Gaussian Splatting inference...")

        try:
            self.pipeline.run()
        except KeyboardInterrupt:
            logger.info("Inference interrupted by user.")
        finally:
            Profiler.print_stats()
