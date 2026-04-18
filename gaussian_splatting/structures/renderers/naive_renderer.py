from dataclasses import dataclass


@dataclass
class NaiveRendererParams:
    width: int
    height: int
    focal_length: float
    near_plane: float = 1e-4
    covariance_regularization: float = 0.3
    gaussian_extent: float = 3.0

    @classmethod
    def from_dict(
        cls,
        config_dict: dict,
    ) -> "NaiveRendererParams":
        if not isinstance(config_dict, dict):
            raise ValueError(f"NaiveRendererParams must be a dictionary, got '{type(config_dict).__name__}'.")

        mandatory_fields = {
            "width",
            "height",
            "focal_length",
        }
        if not set(config_dict.keys()).issuperset(mandatory_fields):
            missing_fields = mandatory_fields - set(config_dict.keys())
            raise ValueError(
                f"NaiveRendererParams is missing the following mandatory fields: {', '.join(missing_fields)}, "
                f"got {', '.join(config_dict.keys())}."
            )

        width = config_dict["width"]
        height = config_dict["height"]
        focal_length = config_dict["focal_length"]
        near_plane = config_dict.get("near_plane", 1e-4)
        covariance_regularization = config_dict.get("covariance_regularization", 0.3)
        gaussian_extent = config_dict.get("gaussian_extent", 3.0)

        return NaiveRendererParams(
            width=width,
            height=height,
            focal_length=focal_length,
            near_plane=near_plane,
            covariance_regularization=covariance_regularization,
            gaussian_extent=gaussian_extent,
        )
