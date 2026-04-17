from dataclasses import dataclass


@dataclass
class ImageShape:
    width: int
    height: int

    @classmethod
    def from_dict(
        cls,
        configuration: dict,
    ) -> "ImageShape":
        if not isinstance(configuration, dict):
            raise ValueError(f"ImageShape configuration must be a dictionary, got '{type(configuration).__name__}'.")

        mandatory_fields = {
            "width",
            "height",
        }

        if not set(configuration.keys()).issuperset(mandatory_fields):
            missing_fields = mandatory_fields - set(configuration.keys())
            raise ValueError(
                f"ImageShape configuration is missing the following mandatory fields: {', '.join(missing_fields)}, "
                f"got {', '.join(configuration.keys())}."
            )

        width = configuration["width"]
        height = configuration["height"]

        if (
            not isinstance(width, int)
            or not isinstance(height, int)
        ):  # fmt:skip
            raise TypeError(
                f"ImageShape 'width' and 'height' should be integers, "
                f"got {type(width).__name__} and {type(height).__name__}."
            )

        return cls(
            width=width,
            height=height,
        )
