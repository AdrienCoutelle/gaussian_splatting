import json
import os

import cv2
import mlx.core as mx
import numpy as np

from gaussian_splatting.structures.camera import Camera


class GaussianSplattingDataset:
    def __init__(
        self,
        images_folder_path: str,
        poses_path: str,
        intrinsics_path: str,
        scale: int = 1,
        validation_index: int = 0,
    ) -> None:
        self.images_folder_path = images_folder_path
        self.scale = scale

        with open(poses_path) as f:
            poses_data: list[dict] = json.load(f)

        with open(intrinsics_path) as f:
            intrinsics_data: list[dict] = json.load(f)

        intrinsics_by_id: dict[int, dict] = {cam["camera_id"]: cam for cam in intrinsics_data}

        items: list[tuple[str, Camera]] = []
        for entry in poses_data:
            intrinsics = intrinsics_by_id[entry["camera_id"]]
            pose = self._compute_pose(
                qw=entry["rotation"]["qw"],
                qx=entry["rotation"]["qx"],
                qy=entry["rotation"]["qy"],
                qz=entry["rotation"]["qz"],
                tx=entry["position"]["x"],
                ty=entry["position"]["y"],
                tz=entry["position"]["z"],
            )

            # Scale camera width, height, and focal length
            width = int(intrinsics["width"] // self.scale)
            height = int(intrinsics["height"] // self.scale)
            focal_length = ((intrinsics["fx"] + intrinsics["fy"]) / 2.0) / self.scale

            image_name = os.path.splitext(entry["name"])[0]
            items.append(
                (
                    image_name,
                    Camera(
                        pose=pose,
                        focal_length=focal_length,
                        width=width,
                        height=height,
                    ),
                )
            )

        if validation_index >= len(items):
            raise ValueError("Wrong index")

        self.items = []
        self.validation_item = None
        for i, item in enumerate(items):
            if i == validation_index:
                image_name, camera = item
                image = self._load_image(image_name, camera.width, camera.height)
                self.validation_item = (image, camera)
            else:
                self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[mx.array, Camera]:
        image_name, camera = self.items[idx]
        image = self._load_image(image_name, camera.width, camera.height)
        return image, camera

    @staticmethod
    def _compute_pose(
        qw: float,
        qx: float,
        qy: float,
        qz: float,
        tx: float,
        ty: float,
        tz: float,
    ) -> mx.array:
        """
        Convert a COLMAP world-to-camera pose to a camera-to-world matrix in the OpenGL convention
        (X right, Y up, -Z forward). COLMAP stores: p_cam = R_cw @ p_world + t_cw.
        """
        q = np.array([qw, qx, qy, qz], dtype=np.float64)
        q /= np.linalg.norm(q)
        w, x, y, z = q

        R_cw = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ],
            dtype=np.float64,
        )
        t_cw = np.array([tx, ty, tz], dtype=np.float64)

        R_wc = R_cw.T
        C_world = -R_cw.T @ t_cw

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R_wc
        pose[:3, 3] = C_world

        # COLMAP convention (Y down, Z forward) → OpenGL convention (Y up, -Z forward)
        flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        pose = pose @ flip

        return mx.array(pose, dtype=mx.float32)

    def _load_image(
        self,
        image_name: str,
        target_width: int,
        target_height: int,
    ) -> mx.array:
        extensions = [".png", ".jpg", ".jpeg"]

        path = None
        for ext in extensions:
            candidate_path = os.path.join(self.images_folder_path, image_name + ext)
            if os.path.exists(candidate_path):
                path = candidate_path
                break

        if path is None:
            raise FileNotFoundError(f"No image found for {image_name} with extensions {extensions}")

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image if the scale factor is not 1
        if self.scale != 1:
            img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

        return mx.array(img / 255.0, dtype=mx.float16)
