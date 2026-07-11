import json
import os
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from gaussian_splatting.structures.dataset import GaussianSplattingDataset
from gaussian_splatting.structures.renderer.renderer import Renderer, RendererConfig
from gaussian_splatting.structures.training.trainer import Trainer, TrainerConfig
from gaussian_splatting.utils.colmap import ColmapConfig, ColmapRunner
from gaussian_splatting.utils.image import is_image
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply.ply_loader import PLYLoader
from gaussian_splatting.utils.profiler import Profiler

logger = Logger("TRAINING_LAUNCHER")


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    training_images_path: str
    poses_json_path: str
    intrinsics_json_path: str
    ply_path: str
    max_sh_degree: int = 1
    scale: float = 1

    trainer_config: TrainerConfig
    output_folder: str


class TrainingLauncher:
    def __init__(
        self,
        configuration: TrainingConfig,
    ) -> None:
        self.configuration = configuration

        if not os.path.exists(self.configuration.training_images_path):
            raise FileNotFoundError(f"Training images folder does not exist: {self.configuration.training_images_path}")

        images = [
            f
            for f in os.listdir(self.configuration.training_images_path)
            if is_image(f)
        ]  # fmt: skip
        if len(images) == 0:
            raise ValueError(f"No image files found in: {self.configuration.training_images_path}")
        logger.info(f"{len(images)} training images found in {self.configuration.training_images_path}.")

        self.run_colmap_if_needed()

        dataset = GaussianSplattingDataset(
            images_folder_path=self.configuration.training_images_path,
            poses_path=self.configuration.poses_json_path,
            intrinsics_path=self.configuration.intrinsics_json_path,
            scale=self.configuration.scale,
        )
        logger.info(f"Dataset created with {len(dataset)} entries.")

        ply_handler = PLYLoader(self.configuration.ply_path)
        ply_handler.log_info()
        gaussian_collection = ply_handler.get_gaussians(max_sh_degree=self.configuration.max_sh_degree)

        renderer = Renderer(self._build_renderer_config())

        init_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(self.configuration.output_folder, init_date)

        self.trainer = Trainer(
            gaussians_collection=gaussian_collection,
            renderer=renderer,
            dataset=dataset,
            output_folder=output_folder,
            configuration=self.configuration.trainer_config,
        )

    def _build_renderer_config(self) -> RendererConfig:
        with open(self.configuration.intrinsics_json_path) as f:
            intrinsics_data: list[dict] = json.load(f)
        intrinsics = intrinsics_data[0]
        scale = self.configuration.scale
        return RendererConfig(
            width=int(intrinsics["width"] // scale),
            height=int(intrinsics["height"] // scale),
            focal_length=((intrinsics["fx"] + intrinsics["fy"]) / 2.0) / scale,
        )

    def run_colmap_if_needed(self) -> None:
        files = {
            "poses": self.configuration.poses_json_path,
            "intrinsics": self.configuration.intrinsics_json_path,
            "ply": self.configuration.ply_path,
        }
        existing = {
            name: os.path.exists(path)
            for name, path in files.items()
        }  # fmt:skip

        if all(existing.values()):
            logger.info("COLMAP outputs already exist, skipping COLMAP.")
            return

        if any(existing.values()):
            missing = [name for name, exists in existing.items() if not exists]
            present = [name for name, exists in existing.items() if exists]
            raise FileNotFoundError(f"Inconsistent COLMAP outputs: {present} exist but {missing} do not.")

        logger.info("COLMAP outputs not found, running COLMAP...")
        colmap_config = ColmapConfig(
            images_path=self.configuration.training_images_path,
            output_folder=os.path.dirname(self.configuration.poses_json_path),
            poses_filename=os.path.basename(self.configuration.poses_json_path),
            intrinsics_filename=os.path.basename(self.configuration.intrinsics_json_path),
            points_filename=os.path.basename(self.configuration.ply_path),
        )
        runner = ColmapRunner(colmap_config)
        runner.run()

    def run(self) -> None:
        logger.info("Starting Gaussian Splatting training...")

        try:
            self.trainer.run()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        finally:
            Profiler.print_stats()
