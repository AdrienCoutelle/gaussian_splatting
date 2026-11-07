import datetime
import math
import os
from typing import Annotated

import cv2
import mlx.core as mx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gaussian_splatting.structures.dataset import GaussianSplattingDataset
from gaussian_splatting.structures.gaussian import Gaussians
from gaussian_splatting.structures.renderer.renderer import Renderer
from gaussian_splatting.structures.training.loss import L1Loss, RenderingLoss
from gaussian_splatting.structures.training.optimizer import GaussianOptimizer, LearningRatesConfig
from gaussian_splatting.utils.differentiability_check import check_renderer_differentiability
from gaussian_splatting.utils.image import stack_images_horizontally
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply.ply_saver import PLYSaver
from gaussian_splatting.utils.profiler import profile

logger = Logger("TRAINER")


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iterations: Annotated[int, Field(gt=0)]
    learning_rates: LearningRatesConfig
    save_every_n_iterations: Annotated[int, Field(gt=0)] = 5000
    render_scale: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    gradient_accumulation_steps: Annotated[int, Field(gt=0)] = 1
    max_gaussians_per_step: int = 0  # 0 = no limit; positive = random subsample per step
    log_every_n_iterations: Annotated[int, Field(gt=0)] = 100
    validate_every_n_iterations: Annotated[int, Field(gt=0)] = 500

    densification_interval: int = 10
    prune_opacity_threshold: float = 0.005
    densify_grad_threshold: float = 0.0002
    split_scale_threshold: float = 0.01


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
            loss=L1Loss(),
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

        self.params = {
            "positions": mx.array(gaussians.positions, dtype=mx.float32),
            "quaternions": mx.array(gaussians.quaternions, dtype=mx.float32),
            "scales": mx.array(gaussians.scales, dtype=mx.float32),
            "sh_coeffs": mx.array(gaussians.sh_coeffs, dtype=mx.float32),
            "opacities": mx.array(gaussians.opacities, dtype=mx.float32),
        }

        self.optimizer = GaussianOptimizer(configuration.learning_rates)

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
                    scalar_value=self._num_gaussians,
                    global_step=iteration,
                )
                running_loss = 0.0

            if iteration % config.save_every_n_iterations == 0:
                self._save_checkpoint(iteration)

        self.tensorboard_writer.close()

    @property
    def _num_gaussians(self) -> int:
        return self.params["positions"].shape[0]

    def _run_iteration(
        self,
        iteration: int,
    ) -> float:
        camera_index = mx.random.randint(0, len(self.dataset)).item()
        gt_image, camera = self.dataset[camera_index]

        loss, grads = self.rendering_loss.compute_value_and_grad(
            params=self.params,
            camera=camera,
            gt_image=gt_image,
        )
        self.params = self.optimizer.step(self.params, grads)

        mx.eval(self.params)

        if iteration % self.configuration.densification_interval == 0:
            stats = self._run_adaptive_density_control(grads)
            for name, count in stats.items():
                self.tensorboard_writer.add_scalar(f"Densification/{name}", count, iteration)

        return loss

    def _run_adaptive_density_control(
        self,
        grads: dict[str, mx.array],
    ) -> dict[str, int]:
        """Prunes transparent Gaussians, then clones small / splits large ones in under-reconstructed areas.

        A Gaussian is under-reconstructed when its view-space positional gradient is large. Small ones are
        cloned (duplicated in place), large ones are split into two smaller ones sampled inside the original.

        Returns how many Gaussians were pruned, cloned and split, and how many remain.
        """
        config = self.configuration
        num_gaussians = self.params["positions"].shape[0]

        should_prune = self.params["opacities"].reshape(-1) < config.prune_opacity_threshold
        should_densify = mx.linalg.norm(grads["positions"], axis=-1) > config.densify_grad_threshold
        is_large = mx.max(mx.exp(self.params["scales"]), axis=-1) > config.split_scale_threshold

        should_split = np.array(should_densify & is_large & ~should_prune)
        should_clone = np.array(should_densify & ~is_large & ~should_prune)
        should_prune = np.array(should_prune)

        # Original Gaussians that survive unchanged, then clones, then the two halves of each split
        keep_idx = np.flatnonzero(~should_prune & ~should_split)
        clone_idx = np.flatnonzero(should_clone)
        split_idx = np.flatnonzero(should_split)
        new_indices = mx.array(np.concatenate([keep_idx, clone_idx, split_idx, split_idx]), dtype=mx.int32)

        new_params = {name: value[new_indices] for name, value in self.params.items()}

        num_split = len(split_idx)
        if num_split > 0:
            split_start = len(keep_idx) + len(clone_idx)
            split_idx = mx.array(split_idx, dtype=mx.int32)
            split_params = self._split_gaussians({name: value[split_idx] for name, value in self.params.items()})
            for name in ("positions", "scales"):
                new_params[name] = mx.concatenate([new_params[name][:split_start], split_params[name]], axis=0)

        self.params = new_params
        self.optimizer.reindex(new_indices, num_gaussians=num_gaussians)
        mx.eval(self.params)

        return {
            "pruned": int(should_prune.sum()),
            "cloned": len(clone_idx),
            "split": num_split,
            "num_gaussians": self._num_gaussians,
        }

    @staticmethod
    def _split_gaussians(
        params: dict[str, mx.array],
        scale_factor: float = 1.6,
    ) -> dict[str, mx.array]:
        scales = mx.exp(params["scales"])
        quaternions = params["quaternions"] / mx.linalg.norm(params["quaternions"], axis=-1, keepdims=True)

        offsets = [_rotate_by_quaternions(mx.random.normal(shape=scales.shape) * scales, quaternions) for _ in range(2)]
        positions = mx.concatenate([params["positions"] + offset for offset in offsets], axis=0)

        child_scales = params["scales"] - math.log(scale_factor)
        return {
            "positions": positions,
            "scales": mx.concatenate([child_scales, child_scales], axis=0),
        }

    def _save_checkpoint(
        self,
        iteration: int | None = None,
    ) -> None:
        gaussians = Gaussians.from_tensors(
            positions=self.params["positions"],
            quaternions=self.params["quaternions"],
            scales=self.params["scales"],
            sh_coeffs=self.params["sh_coeffs"],
            opacities=self.params["opacities"],
        )

        file_name = (
            f"checkpoint_iteration_{iteration}"
            if iteration is not None
            else "checkpoint_final"
        )  # fmt:skip

        ply_saver = PLYSaver(os.path.join(self.output_folder, "checkpoints", f"{file_name}.ply"))
        ply_saver.save_gaussians(gaussians)

    def _run_validation(
        self,
        iteration: int,
    ) -> None:
        gt_image, camera = self.dataset.validation_item

        gaussians = Gaussians.from_tensors(
            positions=self.params["positions"],
            quaternions=self.params["quaternions"],
            scales=self.params["scales"],
            sh_coeffs=self.params["sh_coeffs"],
            opacities=self.params["opacities"],
        )

        t0 = datetime.datetime.now()
        image = self.renderer.render_tensor(
            camera=camera,
            gaussians=gaussians,
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
        self.tensorboard_writer.add_scalar("Stats/num_gaussians", self._num_gaussians, iteration)
        self.tensorboard_writer.add_scalar("Stats/render_time", render_time, iteration)
        self.tensorboard_writer.add_image("Validation/Image", stacked_validation_image, iteration, dataformats="HWC")
        self.tensorboard_writer.flush()

        cv2.imwrite(
            os.path.join(self.validation_images_folder, f"iteration_{iteration:06d}.png"),
            cv2.cvtColor((stacked_validation_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )


def _rotate_by_quaternions(
    vectors: mx.array,
    quaternions: mx.array,
) -> mx.array:
    """Rotates each vector (N, 3) by its unit quaternion (N, 4), stored as (w, x, y, z)."""
    w = quaternions[:, :1]
    xyz = quaternions[:, 1:]
    t = 2.0 * _cross(xyz, vectors)
    return vectors + w * t + _cross(xyz, t)


def _cross(
    a: mx.array,
    b: mx.array,
) -> mx.array:
    return mx.stack(
        [
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ],
        axis=-1,
    )
