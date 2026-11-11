import mlx.core as mx


class Camera:
    def __init__(
        self,
        pose: mx.array,
        focal_length: float,
        width: int,
        height: int,
    ):
        self.pose = pose
        self.focal_length = focal_length
        self.width = width
        self.height = height

    @property
    def f(self) -> float:
        return self.focal_length

    @property
    def h(self) -> int:
        return self.height

    @property
    def w(self) -> int:
        return self.width

    @property
    def principal_point(self) -> tuple[float, float]:
        return (
            self.width / 2.0,
            self.height / 2.0,
        )

    def downscaled(
        self,
        factor: int,
    ) -> "Camera":
        if factor == 1:
            return self
        return Camera(
            pose=self.pose,
            focal_length=self.focal_length / factor,
            width=max(1, round(self.width / factor)),
            height=max(1, round(self.height / factor)),
        )
