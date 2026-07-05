from pydantic import BaseModel, ConfigDict

from gaussian_splatting.structures.inference_pipelines.factory import InferencePipelineFactory, PipelineConfig
from gaussian_splatting.structures.renderer.renderer import Renderer, RendererConfig
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply.ply_loader import PLYLoader
from gaussian_splatting.utils.profiler import Profiler

logger = Logger("INFERENCE_LAUNCHER")


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ply_file_path: str
    renderer_config: RendererConfig
    inference_pipeline_config: PipelineConfig
    output_folder: str


class InferenceLauncher:
    def __init__(
        self,
        config: InferenceConfig,
    ):
        self.config = config

        ply_handler = PLYLoader(file_path=self.config.ply_file_path)
        gaussian_collection = ply_handler.get_gaussians()

        logger.info(f"Loaded {len(gaussian_collection.positions)} gaussians from PLY file.")

        renderer = Renderer(self.config.renderer_config)

        self.pipeline = InferencePipelineFactory.create(
            renderer=renderer,
            gaussians=gaussian_collection.to_list(),
            configuration=self.config.inference_pipeline_config,
            output_folder=self.config.output_folder,
        )

    def run(self) -> None:
        logger.info("Starting Gaussian Splatting inference...")

        try:
            self.pipeline.run()
        except KeyboardInterrupt:
            logger.info("Inference interrupted by user.")
        finally:
            Profiler.print_stats()
