import os
from typing import Annotated

import cv2
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.dataset import GaussianSplattingDataset
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply.ply_saver import PLYSaver
from gaussian_splatting.utils.profiler import profile

logger = Logger("TRAINER")


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: Annotated[int, Field(gt=0)]
    learning_rate: float = 1e-3
    save_every_n_epochs: int = 10
    render_scale: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    gradient_accumulation_steps: Annotated[int, Field(gt=0)] = 1


@profile
class Trainer:
    def __init__(
        self,
        gaussians_collection: GaussianCollection,
        renderer: BaseRenderer,
        dataset: GaussianSplattingDataset,
        output_folder: str,
        configuration: TrainerConfig,
        device: torch.device,
    ) -> None:
        self.renderer = renderer
        self.dataset = dataset
        self.output_folder = output_folder
        self.configuration = configuration
        self.device = device

        os.makedirs(output_folder, exist_ok=True)

        # Wrap Gaussian tensors as trainable parameters
        self.positions = nn.Parameter(gaussians_collection.positions.to(device=device, dtype=torch.float32))
        self.quaternions = nn.Parameter(gaussians_collection.quaternions.to(device=device, dtype=torch.float32))
        self.scales = nn.Parameter(gaussians_collection.scales.to(device=device, dtype=torch.float32))
        self.sh_coeffs = nn.Parameter(gaussians_collection.sh_coeffs.to(device=device, dtype=torch.float32))
        self.opacities = nn.Parameter(gaussians_collection.opacities.to(device=device, dtype=torch.float32))

        self.optimizer = torch.optim.Adam(
            [self.positions, self.quaternions, self.scales, self.sh_coeffs, self.opacities],
            lr=configuration.learning_rate,
        )

    def _build_gaussian_collection(self) -> GaussianCollection:
        return GaussianCollection.from_tensors(
            positions=self.positions,
            quaternions=self.quaternions,
            scales=self.scales,
            sh_coeffs=self.sh_coeffs,
            opacities=self.opacities,
        )

    def _clear_device_cache(self) -> None:
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _load_training_image(
        self,
        image_name: str,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor | None:
        """Load a training image as a float tensor (H, W, 3) in [0, 1] resized to the camera resolution."""
        for ext in [".png", ".jpg", ".jpeg"]:
            path = os.path.join(self.dataset.images_folder_path, image_name + ext)
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != target_height or img.shape[1] != target_width:
                img = cv2.resize(img, (target_width, target_height))
            return torch.tensor(img / 255.0, dtype=torch.float32, device=self.device)
        return None

    def _save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        checkpoint_path = os.path.join(self.output_folder, f"checkpoint_epoch_{epoch:04d}.ply")
        gaussians_collection = GaussianCollection.from_tensors(
            positions=self.positions.detach(),
            quaternions=self.quaternions.detach(),
            scales=self.scales.detach(),
            sh_coeffs=self.sh_coeffs.detach(),
            opacities=self.opacities.detach(),
        )
        PLYSaver(checkpoint_path).save_gaussians(gaussians_collection)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def run(self) -> None:
        logger.info(f"Starting training for {self.configuration.epochs} epochs over {len(self.dataset)} cameras.")

        camera_items = list(self.dataset.items)
        scale = self.configuration.render_scale

        for epoch in tqdm(range(self.configuration.epochs), desc="Training"):
            epoch_loss = 0.0
            num_rendered = 0

            # Shuffle cameras each epoch for better gradient diversity
            indices = torch.randperm(len(camera_items)).tolist()
            accum_steps = self.configuration.gradient_accumulation_steps

            for step, idx in enumerate(indices):
                logger.info(
                    f"Epoch {epoch + 1}/{self.configuration.epochs} — rendering camera {idx + 1}/{len(camera_items)}"
                )
                image_name, camera = camera_items[idx]

                render_h = max(1, round(camera.h * scale))
                render_w = max(1, round(camera.w * scale))
                render_camera = Camera(
                    pose=camera.pose,
                    focal_length=camera.focal_length * scale,
                    width=render_w,
                    height=render_h,
                )

                gt_image = self._load_training_image(
                    image_name=image_name,
                    target_height=render_h,
                    target_width=render_w,
                )
                if gt_image is None:
                    continue

                # Build collection directly from parameter tensors — no per-gaussian indexing
                gaussians = self._build_gaussian_collection()
                rendered = self.renderer.render_tensor(camera=render_camera, gaussians=gaussians)

                loss = torch.mean(torch.abs(rendered - gt_image)) / accum_steps

                logger.info(
                    f"Epoch {epoch + 1}/{self.configuration.epochs} — camera {idx + 1}/{len(camera_items)} — "
                    f"L1 loss: {loss.item() * accum_steps:.6f}"
                )

                if step % accum_steps == 0:
                    self.optimizer.zero_grad()

                loss.backward()

                is_last_step = step == len(indices) - 1
                if (step + 1) % accum_steps == 0 or is_last_step:
                    self.optimizer.step()
                    logger.info("Ran backpropagation and optimizer step.")

                epoch_loss += loss.item()
                num_rendered += 1

                del rendered, gaussians, loss
                self._clear_device_cache()

                logger.info(
                    f"Epoch {epoch + 1}/{self.configuration.epochs} — camera {idx + 1}/{len(camera_items)} — "
                    f"rendered and updated Gaussians, cleared cache."
                )

            if num_rendered > 0:
                avg_loss = epoch_loss / num_rendered
                logger.info(f"Epoch {epoch + 1}/{self.configuration.epochs} — avg L1 loss: {avg_loss:.6f}")

            if (epoch + 1) % self.configuration.save_every_n_epochs == 0:
                self._save_checkpoint(epoch + 1)

            logger.info(f"Completed epoch {epoch + 1}/{self.configuration.epochs}.")

        self._save_checkpoint(self.configuration.epochs)
