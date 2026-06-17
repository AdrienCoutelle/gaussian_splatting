import json
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer
from gaussian_splatting.utils.logger import Logger

logger = Logger("TRAINER")


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: Annotated[int, Field(gt=0)]
    # paths...
    # output folder...


class Trainer:
    def __init__(
        self,
        gaussians_collection: GaussianCollection,
        renderer: BaseRenderer,
        training_images_path: str,
        poses_json_path: str,
        intrinsics_json_path: str,
        output_folder: str,
        configuration: TrainerConfig,
        device: torch.device,
    ) -> None:
        self.renderer = renderer
        self.configuration = configuration
        self.device = device

        self.training_images_path = Path(training_images_path)
        if not self.training_images_path.exists() or not self.training_images_path.is_dir():
            raise FileNotFoundError(
                "GaussianSplattingTrainer 'training_images_path' must point to an existing image directory, "
                f"got '{training_images_path}'."
            )

        self.poses_json_path = Path(poses_json_path)
        if not self.poses_json_path.exists():
            raise FileNotFoundError(f"Poses json file does not exist: '{poses_json_path}'.")

        self.intrinsics_json_path = Path(intrinsics_json_path)
        if not self.intrinsics_json_path.exists():
            raise FileNotFoundError(f"Intrinsics json file does not exist: '{intrinsics_json_path}'.")

        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.models_folder = self.output_folder / "models"
        self.models_folder.mkdir(parents=True, exist_ok=True)
        self.previews_folder = self.output_folder / "previews"
        self.previews_folder.mkdir(parents=True, exist_ok=True)

        self.training_views = self._load_training_views(
            poses_json_path=self.poses_json_path,
            intrinsics_json_path=self.intrinsics_json_path,
        )

        if len(self.training_views) == 0:
            raise ValueError("No valid training views were loaded.")

        # means = initial_gaussians.means.to(device=self.device, dtype=torch.float32)
        # covariances = initial_gaussians.covariances.to(device=self.device, dtype=torch.float32)
        # sh_coeffs = initial_gaussians.sh_coeffs.to(device=self.device, dtype=torch.float32)
        # opacities = initial_gaussians.opacities.to(device=self.device, dtype=torch.float32)

        # sigma_init = torch.sqrt(torch.clamp(covariances.diagonal(dim1=1, dim2=2), min=1e-12))
        # self.means = torch.nn.Parameter(means)
        # self.log_scales = torch.nn.Parameter(torch.log(torch.clamp(sigma_init, min=1e-4)))
        # # SH DC coefficients are trained directly without activation (unbounded)
        # self.sh_coeffs = torch.nn.Parameter(sh_coeffs)  # (N, num_coeffs, 3)
        # self.opacity_logits = torch.nn.Parameter(torch.logit(torch.clamp(opacities, min=1e-4, max=1.0 - 1e-4)))

        self.optimizer = torch.optim.Adam(
            [
                self.means,
                self.log_scales,
                self.sh_coeffs,
                self.opacity_logits,
            ],
            lr=self.configuration.learning_rate,
        )

    def run(self) -> None:
        logger.info(f"Starting training for {self.configuration.max_steps} steps.")

        for step in range(1, self.configuration.max_steps + 1):
            view_index = (step - 1) % len(self.training_views)
            camera, target_image = self.training_views[view_index]

            self.optimizer.zero_grad(set_to_none=True)

            gaussians = self._current_gaussians().to_list()
            rendered_image = self.renderer.render_tensor(
                camera=camera,
                gaussians=gaussians,
            )

            loss = torch.mean((rendered_image - target_image) ** 2)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.log_scales.clamp_(min=np.log(1e-4), max=np.log(0.5))

            if step == 1 or step % self.configuration.log_every_n_steps == 0:
                logger.info(f"Step {step:06d}/{self.configuration.max_steps:06d} - mse={loss.item():.6f}")

            if step == 1 or step % self.configuration.preview_every_n_steps == 0:
                self._save_preview(
                    step=step,
                    camera=camera,
                )

            if step == 1 or step % self.configuration.checkpoint_every_n_steps == 0:
                self._save_checkpoint(step=step)

        self._save_checkpoint(step=self.configuration.max_steps, filename="final.pt")
        logger.info(f"Training completed. Artifacts saved to '{self.output_folder}'.")

    def _current_gaussians(self) -> GaussianCollection:
        scales = torch.exp(self.log_scales)
        # Identity quaternions (no rotation): [w=1, x=0, y=0, z=0]
        quaternions = torch.zeros((self.means.shape[0], 4), device=self.device, dtype=torch.float32)
        quaternions[:, 0] = 1.0
        opacities = torch.sigmoid(self.opacity_logits)

        return GaussianCollection.from_tensors(
            positions=self.means,
            quaternions=quaternions,
            scales=scales,
            sh_coeffs=self.sh_coeffs,
            opacities=opacities,
        )

    def _save_preview(
        self,
        step: int,
        camera: Camera,
    ) -> None:
        with torch.no_grad():
            gaussians = self._current_gaussians().to_list()
            rendered = self.renderer.render(
                camera=camera,
                gaussians=gaussians,
            ).array

        preview_path = self.previews_folder / f"step_{step:06d}.png"
        preview_bgr = (np.clip(rendered, 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(str(preview_path), cv2.cvtColor(preview_bgr, cv2.COLOR_RGB2BGR))

    def _save_checkpoint(
        self,
        step: int,
        filename: str | None = None,
    ) -> None:
        checkpoint_name = filename or f"step_{step:06d}.pt"
        checkpoint_path = self.models_folder / checkpoint_name

        with torch.no_grad():
            collection = self._current_gaussians()
            payload = {
                "step": step,
                "means": collection.positions.detach().cpu(),
                "quaternions": collection.quaternions.detach().cpu(),
                "scales": collection.scales.detach().cpu(),
                "sh_coeffs": collection.sh_coeffs.detach().cpu(),
                "opacities": collection.opacities.detach().cpu(),
            }

        torch.save(payload, checkpoint_path)

    def _load_training_views(
        self,
        poses_json_path: Path,
        intrinsics_json_path: Path,
    ) -> list[tuple[Camera, torch.Tensor]]:
        with open(intrinsics_json_path) as file:
            intrinsics = json.load(file)
        with open(poses_json_path) as file:
            poses = json.load(file)

        if not isinstance(intrinsics, list) or len(intrinsics) == 0:
            raise ValueError("Intrinsics json must be a non-empty list.")
        if not isinstance(poses, list):
            raise ValueError("Poses json must be a list.")

        intrinsics_by_camera_id = {int(camera["camera_id"]): camera for camera in intrinsics}

        views: list[tuple[Camera, torch.Tensor]] = []
        render_width = int(self.renderer.config.width)
        render_height = int(self.renderer.config.height)

        for pose_entry in poses:
            image_name = pose_entry["name"]
            image_path = self.training_images_path / image_name
            if not image_path.exists():
                logger.warning(f"Skipping view '{image_name}': image does not exist.")
                continue

            camera_id = int(pose_entry["camera_id"])
            if camera_id not in intrinsics_by_camera_id:
                logger.warning(f"Skipping view '{image_name}': unknown camera_id '{camera_id}'.")
                continue

            camera_intrinsics = intrinsics_by_camera_id[camera_id]
            world_to_camera_rotation = _quaternion_to_rotation_matrix(
                qw=float(pose_entry["rotation"]["qw"]),
                qx=float(pose_entry["rotation"]["qx"]),
                qy=float(pose_entry["rotation"]["qy"]),
                qz=float(pose_entry["rotation"]["qz"]),
            )

            translation = np.array(
                [
                    float(pose_entry["position"]["x"]),
                    float(pose_entry["position"]["y"]),
                    float(pose_entry["position"]["z"]),
                ],
                dtype=np.float32,
            )

            camera_to_world_rotation = world_to_camera_rotation.transpose()
            camera_center = -camera_to_world_rotation @ translation

            camera_pose = torch.eye(4, dtype=torch.float32, device=self.device)
            camera_pose[:3, :3] = torch.from_numpy(camera_to_world_rotation).to(self.device)
            camera_pose[:3, 3] = torch.from_numpy(camera_center).to(self.device)

            focal_length = float(camera_intrinsics.get("fx", camera_intrinsics.get("f")))
            if "fy" in camera_intrinsics:
                focal_length = 0.5 * (focal_length + float(camera_intrinsics["fy"]))

            camera = Camera(
                pose=camera_pose,
                focal_length=focal_length,
                width=render_width,
                height=render_height,
            )

            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                logger.warning(f"Skipping view '{image_name}': unable to read image.")
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                image_rgb,
                (render_width, render_height),
                interpolation=cv2.INTER_AREA,
            )
            image_tensor = torch.from_numpy(resized).to(device=self.device, dtype=torch.float32) / 255.0

            views.append((camera, image_tensor))

        return views


def _quaternion_to_rotation_matrix(
    qw: float,
    qx: float,
    qy: float,
    qz: float,
) -> np.ndarray:
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        raise ValueError("Invalid zero-norm quaternion.")

    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm

    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )
