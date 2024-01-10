from PIL import Image
from typing import Dict
from typing import Tuple
from pydantic import Field
from pydantic import BaseModel
from cftool.cv import to_rgb
from cftool.cv import to_uint8

from .common import APIs
from .common import sd_img2img_name
from .common import register_sd
from .common import get_sd_from
from .common import handle_diffusion_model
from .common import handle_diffusion_hooks
from .common import get_normalized_arr_from_diffusion
from .common import Img2ImgDiffusionModel
from ..schema import IImageNode
from ...core import Node
from ...core import Schema


class Img2ImgSDSettings(BaseModel):
    text: str = Field(..., description="The text that we want to handle.")
    wh: Tuple[int, int] = Field(
        (0, 0),
        description="The output size, `0` means as-is",
    )
    fidelity: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="The fidelity of the input image.",
    )
    keep_alpha: bool = Field(
        True,
        description="""
Whether the returned image should keep the alpha-channel of the input image or not.
> If the input image is a sketch image, then `keep_alpha` needs to be False in most of the time.  
""",
    )


class Img2ImgSDModel(Img2ImgDiffusionModel, Img2ImgSDSettings):
    pass


@Node.register(sd_img2img_name)
class SDImg2ImgNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = Img2ImgSDModel
        schema.description = "Use Stable Diffusion to perform image to image."
        return schema

    @classmethod
    async def warmup(cls) -> None:
        register_sd()

    async def execute(self) -> Dict[str, Image.Image]:
        data = Img2ImgSDModel(**self.data)
        image = await self.get_image_from("url")
        if not data.keep_alpha:
            image = to_rgb(image)
        w, h = data.wh
        if w > 0 and h > 0:
            image = image.resize((w, h), Image.LANCZOS)
        m = get_sd_from(APIs.SD, data)  # type: ignore
        kwargs = handle_diffusion_model(m, data)
        await handle_diffusion_hooks(m, data, self, kwargs)
        img_arr = m.img2img(
            image,
            cond=[data.text],
            max_wh=data.max_wh,
            fidelity=data.fidelity,
            anchor=64,
            **kwargs,
        ).numpy()[0]
        image = Image.fromarray(to_uint8(get_normalized_arr_from_diffusion(img_arr)))
        return {"image": image}


__all__ = [
    "Img2ImgSDModel",
    "SDImg2ImgNode",
]
