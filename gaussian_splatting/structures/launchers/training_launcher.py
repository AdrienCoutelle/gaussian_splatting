from pydantic import BaseModel, ConfigDict

from gaussian_splatting.structures.device import Device
from gaussian_splatting.structures.inference_pipelines.factory import InferencePipelineFactory, PipelineConfig
from gaussian_splatting.structures.renderers.factory import RendererConfig, RendererFactory
from gaussian_splatting.structures.training.trainer import TrainerConfig
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply.ply_loader import PLYLoader

logger = Logger("TRAINING_LAUNCHER")


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inference_pipeline_config: PipelineConfig
    renderer_config: RendererConfig
    trainer_config: TrainerConfig
    output_folder: str


class TrainingLauncher:
    def __init__(
        self,
        configuration: TrainingConfig,
    ) -> None:
        self.configuration = configuration
        device = Device.get()

        ply_handler = PLYLoader("output/colmap/lego/points3D.ply")

        ply_handler.log_info()
        gaussian_collection = ply_handler.get_gaussians()
        gaussians = gaussian_collection.to_list()

        renderer = RendererFactory.create_renderer(
            configuration=self.configuration.renderer_config,
            device=device,
        )

        self.pipeline = InferencePipelineFactory.create(
            renderer=renderer,
            gaussians=gaussians,
            configuration=self.configuration.inference_pipeline_config,
            device=device,
            output_folder=self.configuration.output_folder,
        )

        self.pipeline.run()

    #     self.trainer = Trainer(
    #         initial_gaussians=gaussian_collection,
    #         renderer=renderer,
    #         # training_images_path=self.configuration.training_images_path,
    #         # poses_json_path=self.configuration.poses_json_path,
    #         # intrinsics_json_path=self.configuration.intrinsics_json_path,
    #         output_folder=self.configuration.output_folder,
    #         configuration=self.configuration.training_config,
    #         device=device,
    #     )

    # def run(self) -> None:
    #     logger.info("Starting Gaussian Splatting training...")

    #     try:
    #         self.trainer.run()
    #     except KeyboardInterrupt:
    #         logger.info("Training interrupted by user.")
    #     finally:
    #         Profiler.print_stats()
