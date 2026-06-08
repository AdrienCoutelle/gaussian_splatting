from pydantic import BaseModel, ConfigDict

from gaussian_splatting.structures.renderers.factory import RendererConfig
from gaussian_splatting.structures.training.trainer import TrainerConfig
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply_handler import PLYHandler

logger = Logger("TRAINING_LAUNCHER")


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    renderer_config: RendererConfig
    trainer_config: TrainerConfig
    output_folder: str


class TrainingLauncher:
    def __init__(
        self,
        configuration: TrainingConfig,
    ) -> None:
        self.configuration = configuration
        # device = Device.get()

        ply_handler = PLYHandler("output/colmap/lego/points3D.ply")

        ply_handler.log_info()

        # gaussian_collection = ply_handler.get_gaussians()

        # renderer = RendererFactory.create_renderer(
        #     configuration=self.configuration.renderer_config,
        #     device=device,
        # )

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
