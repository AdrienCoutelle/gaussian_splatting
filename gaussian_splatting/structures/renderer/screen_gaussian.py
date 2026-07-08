from dataclasses import dataclass

import mlx.core as mx


@dataclass
class ScreenSpaceGaussians:
    means_2d: mx.array
    covariances_2d: mx.array
    depths: mx.array
    colors: mx.array
    opacities: mx.array

    def __len__(self) -> int:
        return int(self.means_2d.shape[0])

    def __getitem__(self, idx):
        return ScreenSpaceGaussians(
            means_2d=self.means_2d[idx],
            covariances_2d=self.covariances_2d[idx],
            depths=self.depths[idx],
            colors=self.colors[idx],
            opacities=self.opacities[idx],
        )
