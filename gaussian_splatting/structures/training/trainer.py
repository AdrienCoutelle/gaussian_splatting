import os
from typing import Annotated

import mlx.core as mx
import mlx.optimizers as opt
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from gaussian_splatting.structures.dataset import GaussianSplattingDataset
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderer.renderer import Renderer
from gaussian_splatting.structures.training.utils import ssim
from gaussian_splatting.utils.differentiability_check import check_renderer_differentiability
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.profiler import profile
from gaussian_splatting.utils.tensorboard import TensorBoardWriter

logger = Logger("TRAINER")


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: Annotated[int, Field(gt=0)]
    learning_rate: float = 1e-3
    save_every_n_epochs: int = 10
    render_scale: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    gradient_accumulation_steps: Annotated[int, Field(gt=0)] = 1
    max_gaussians_per_step: int = 0  # 0 = no limit; positive = random subsample per step
    log_every_n_epochs: int = 5  # log metrics + image to TensorBoard every N epochs


@profile
class Trainer:
    def __init__(
        self,
        gaussians_collection: GaussianCollection,
        renderer: Renderer,
        dataset: GaussianSplattingDataset,
        output_folder: str,
        configuration: TrainerConfig,
    ) -> None:
        self.renderer = renderer
        self.dataset = dataset
        self.output_folder = output_folder
        self.configuration = configuration

        check_renderer_differentiability(self.renderer)

        os.makedirs(output_folder, exist_ok=True)

        tb_log_dir = os.path.join(output_folder, "tensorboard")
        self.tensorboard_writer = TensorBoardWriter(tb_log_dir)
        logger.info(f"TensorBoard logs: {tb_log_dir}")

        self.params = {
            "positions": mx.array(gaussians_collection.positions, dtype=mx.float32),
            "quaternions": mx.array(gaussians_collection.quaternions, dtype=mx.float32),
            "scales": mx.array(gaussians_collection.scales, dtype=mx.float32),
            "sh_coeffs": mx.array(gaussians_collection.sh_coeffs, dtype=mx.float32),
            "opacities": mx.array(gaussians_collection.opacities, dtype=mx.float32),
        }

        self.optimizer = opt.Adam(learning_rate=configuration.learning_rate)

    def run(self) -> None:
        logger.info(f"Starting training for {self.configuration.epochs} epochs over {len(self.dataset)} cameras.")

        for epoch in tqdm(range(1, self.configuration.epochs + 1), desc="Training"):
            avg_loss = self._run_epoch(epoch)

            logger.info(f"Epoch {epoch}/{self.configuration.epochs} completed — Avg Loss: {avg_loss:.6f}")
            self.tensorboard_writer.log_scalar("Loss/train_epoch", avg_loss, epoch)

        self.tensorboard_writer.close()

    def _run_epoch(
        self,
        epoch: int,
    ) -> float:
        indices = mx.random.permutation(len(self.dataset)).tolist()

        epoch_loss = 0.0
        num_rendered = 0

        def loss_fn(params, camera, gt_image):
            gaussians = self._build_gaussian_collection(params)
            image = self.renderer.render_tensor(
                camera=camera,
                gaussians=gaussians,
            )
            return self._loss_fn(image, gt_image)

        loss_and_grad_fn = mx.value_and_grad(loss_fn)

        for step, idx in enumerate(indices):
            logger.info(
                f"Epoch {epoch}, Step {step + 1}/{len(self.dataset)}: Rendering and computing loss for image {idx}."
            )
            gt_image, camera = self.dataset[idx]

            loss, grads = loss_and_grad_fn(self.params, camera, gt_image)

            mx.eval(loss)

            logger.info(f"Epoch {epoch}, Step {step + 1}/{len(self.dataset)}: Loss = {loss.item():.6f}")
            epoch_loss += loss.item()
            num_rendered += 1

            self.optimizer.update(self.params, grads)
            mx.eval(self.params, self.optimizer.state)

            # infos on grads
            logger.info(f"Gradients: {[g.mean().item() for g in grads.values()]}")

        return epoch_loss / num_rendered if num_rendered > 0 else 0.0

    def _loss_fn(
        self,
        image: mx.array,
        gt_image: mx.array,
    ) -> mx.array:
        l1 = mx.mean(mx.abs(image - gt_image))
        ssim_loss = 1.0 - ssim(image, gt_image)

        return 0.8 * l1 + 0.2 * ssim_loss

    def _build_gaussian_collection(
        self,
        params: dict[str, mx.array],
    ) -> GaussianCollection:
        return GaussianCollection.from_tensors(
            positions=params["positions"],
            quaternions=params["quaternions"],
            scales=params["scales"],
            sh_coeffs=params["sh_coeffs"],
            opacities=params["opacities"],
        )
