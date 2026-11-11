import datetime
import math
import os
from typing import Annotated

import cv2
import mlx.core as mx
import numpy as np
from pydantic import Field
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gaussian_splatting.structures.dataset import GaussianSplattingDataset
from gaussian_splatting.structures.gaussian import Gaussians
from gaussian_splatting.structures.renderer.renderer import Renderer
from gaussian_splatting.utils.differentiability_check import check_renderer_differentiability
from gaussian_splatting.utils.image import stack_images_horizontally
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply.ply_saver import PLYSaver
from gaussian_splatting.utils.profiler import profile

from .gaussians_training_manager import GaussiansTrainingManager, GaussiansTrainingManagerConfig
from .loss import L1DSSIMLoss, RenderingLoss

logger = Logger("TRAINER")


class TrainerConfig(GaussiansTrainingManagerConfig):
    """Training schedule and logging settings, on top of the Gaussians manager settings (kept flat for the YAML)."""

    iterations: Annotated[int, Field(gt=0)]
    save_every_n_iterations: Annotated[int, Field(gt=0)] = 5000
    render_scale: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    gradient_accumulation_steps: Annotated[int, Field(gt=0)] = 1
    max_gaussians_per_step: int = 0  # 0 = no limit; positive = random subsample per step
    log_every_n_iterations: Annotated[int, Field(gt=0)] = 100
    validate_every_n_iterations: Annotated[int, Field(gt=0)] = 500

    lambda_dssim: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2

    densification_interval: Annotated[int, Field(gt=0)] = 100
    densify_from_iteration: Annotated[int, Field(ge=0)] = 500
    densify_until_iteration: Annotated[int, Field(ge=0)] = 15000  # no densification nor opacity reset from here on


@profile
class Trainer:
    def __init__(
        self,
        gaussians: Gaussians,
        renderer: Renderer,
        dataset: GaussianSplattingDataset,
        output_folder: str,
        configuration: TrainerConfig,
    ) -> None:
        self.renderer = renderer
        self.rendering_loss = RenderingLoss(
            renderer=renderer,
            loss=L1DSSIMLoss(lambda_dssim=configuration.lambda_dssim),
        )
        self.dataset = dataset
        self.output_folder = output_folder
        self.configuration = configuration

        check_renderer_differentiability(self.renderer)

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, "checkpoints"))
        self.validation_images_folder = os.path.join(output_folder, "validation")
        os.makedirs(self.validation_images_folder)

        tb_log_dir = os.path.join(output_folder, "tensorboard")
        self.tensorboard_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logs: {tb_log_dir}")

        self.gaussians_manager = GaussiansTrainingManager(
            gaussians=gaussians,
            cameras=dataset.cameras,
            configuration=configuration,
        )

    def run(self) -> None:
        config = self.configuration
        logger.info(f"Starting training for {config.iterations} iterations over {len(self.dataset)} cameras.")

        running_loss = 0.0
        self._run_validation(iteration=0)

        for iteration in tqdm(range(1, config.iterations + 1), desc="Training"):
            running_loss += self._run_iteration(iteration)

            if iteration % config.log_every_n_iterations == 0:
                self.tensorboard_writer.add_scalar(
                    tag="Loss/train",
                    scalar_value=running_loss / config.log_every_n_iterations,
                    global_step=iteration,
                )
                self.tensorboard_writer.add_scalar(
                    tag="Stats/num_gaussians",
                    scalar_value=self.gaussians_manager.num_gaussians,
                    global_step=iteration,
                )
                running_loss = 0.0

            if iteration % config.validate_every_n_iterations == 0:
                self._run_validation(iteration)

            if iteration % config.save_every_n_iterations == 0:
                self._save_checkpoint(iteration)

        self.tensorboard_writer.close()

    def _run_iteration(
        self,
        iteration: int,
    ) -> float:
        camera_index = mx.random.randint(0, len(self.dataset)).item()
        gt_image, camera = self.dataset[camera_index]

        output = self.rendering_loss.compute_value_and_grad(
            params=self.gaussians_manager.params,
            camera=camera,
            gt_image=gt_image,
        )
        self.gaussians_manager.step(output.grads)

        config = self.configuration
        if iteration < config.densify_until_iteration:
            self.gaussians_manager.accumulate_densification_stats(output.view_space_grads, output.visibility, camera)

            if iteration > config.densify_from_iteration and iteration % config.densification_interval == 0:
                stats = self.gaussians_manager.densify_and_prune(iteration)
                for name, count in stats.items():
                    self.tensorboard_writer.add_scalar(f"Densification/{name}", count, iteration)

            if iteration % config.opacity_reset_interval == 0:
                self.gaussians_manager.reset_opacities()

        return output.loss

    def _save_checkpoint(
        self,
        iteration: int | None = None,
    ) -> None:
        file_name = (
            f"checkpoint_iteration_{iteration}"
            if iteration is not None
            else "checkpoint_final"
        )  # fmt:skip

        ply_saver = PLYSaver(os.path.join(self.output_folder, "checkpoints", f"{file_name}.ply"))
        ply_saver.save_gaussians(self.gaussians_manager.to_gaussians())

    def _run_validation(
        self,
        iteration: int,
    ) -> None:
        gt_image, camera = self.dataset.validation_item

        t0 = datetime.datetime.now()
        image = self.renderer.render_tensor(
            camera=camera,
            gaussians=self.gaussians_manager.to_gaussians(),
        )
        mx.eval(image)
        render_time = (datetime.datetime.now() - t0).total_seconds()

        val_loss = self.rendering_loss.evaluate(image, gt_image)
        mse = mx.mean((mx.clip(image, 0.0, 1.0) - gt_image) ** 2).item()
        psnr = -10.0 * math.log10(max(mse, 1e-10))

        # Ground truth on the left, render on the right
        stacked_validation_image = np.clip(
            stack_images_horizontally(left_image=np.array(gt_image), right_image=np.array(image)), 0.0, 1.0
        )

        self.tensorboard_writer.add_scalar("Loss/validation", val_loss, iteration)
        self.tensorboard_writer.add_scalar("Metrics/validation_psnr", psnr, iteration)
        self.tensorboard_writer.add_scalar("Stats/num_gaussians", self.gaussians_manager.num_gaussians, iteration)
        self.tensorboard_writer.add_scalar("Stats/render_time", render_time, iteration)
        self.tensorboard_writer.add_image("Validation/Image", stacked_validation_image, iteration, dataformats="HWC")
        self.tensorboard_writer.flush()

        cv2.imwrite(
            os.path.join(self.validation_images_folder, f"iteration_{iteration:06d}.png"),
            cv2.cvtColor((stacked_validation_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
