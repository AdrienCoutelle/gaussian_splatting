import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from gaussian_splatting.structures.camera import Camera


class GaussianSplattingDataset(Dataset):
    def __init__(
        self,
        images_folder_path: str,
        poses_path: str,
        intrinsics_path: str,
    ) -> None:
        self.images_folder_path = images_folder_path

        with open(poses_path) as f:
            poses_data: list[dict] = json.load(f)

        with open(intrinsics_path) as f:
            intrinsics_data: list[dict] = json.load(f)

        intrinsics_by_id: dict[int, dict] = {cam["camera_id"]: cam for cam in intrinsics_data}

        self.items: list[tuple[str, Camera]] = []
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
            focal_length = (intrinsics["fx"] + intrinsics["fy"]) / 2.0
            image_name = os.path.splitext(entry["name"])[0]
            self.items.append(
                (
                    image_name,
                    Camera(
                        pose=pose,
                        focal_length=focal_length,
                        width=intrinsics["width"],
                        height=intrinsics["height"],
                    ),
                )
            )

    @staticmethod
    def _compute_pose(
        qw: float,
        qx: float,
        qy: float,
        qz: float,
        tx: float,
        ty: float,
        tz: float,
    ) -> torch.Tensor:
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

        return torch.tensor(pose, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, Camera]:
        return self.items[idx]
