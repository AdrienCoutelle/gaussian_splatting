from pydantic import BaseModel, ConfigDict

from gaussian_splatting.structures.device import Device
from gaussian_splatting.structures.inference_pipelines.factory import InferencePipelineFactory, PipelineConfig
from gaussian_splatting.structures.renderers.factory import RendererConfig, RendererFactory
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
        self.device = Device.get()

        ply_handler = PLYLoader(file_path=self.config.ply_file_path)
        gaussian_collection = ply_handler.get_gaussians()
        gaussians = gaussian_collection.to_list()

        renderer = RendererFactory.create_renderer(
            configuration=self.config.renderer_config,
            device=self.device,
        )

        self.pipeline = InferencePipelineFactory.create(
            renderer=renderer,
            gaussians=gaussians,
            configuration=self.config.inference_pipeline_config,
            device=self.device,
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
