from PIL import Image
from enum import Enum
from typing import Any
from typing import Union
from pydantic import Field
from pydantic import BaseModel
from pydantic_core import core_schema


class DocEnum(Enum):
    """A class that tells use to include it in the documentation"""


class DocModel(BaseModel):
    """A class that tells use to include it in the documentation"""


class ImageField(Image.Image):
    @classmethod
    def __get_pydantic_core_schema__(cls, *args: Any) -> core_schema.CoreSchema:
        return core_schema.with_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v: Any, info: core_schema.ValidationInfo) -> Image.Image:
        if isinstance(v, Image.Image):
            return v
        raise ValueError("Value must be a PIL Image")


TImage = Union[str, ImageField]


class ImageModel(DocModel):
    url: TImage = Field(..., description="The url / PIL.Image instance of the image.")


class ImageAPIOuput(DocModel):
    image: str = Field(..., description="The base64 encoded image.")


class EmptyOutput(BaseModel):
    pass


__all__ = [
    "DocEnum",
    "DocModel",
    "TextModel",
    "TImage",
    "ImageModel",
    "ImageAPIOuput",
    "EmptyOutput",
]
