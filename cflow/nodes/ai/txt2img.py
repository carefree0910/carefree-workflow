from PIL import Image
from typing import Dict
from cftool.cv import to_uint8

from .common import APIs
from .common import sd_txt2img_name
from .common import register_sd
from .common import get_sd_from
from .common import handle_diffusion_model
from .common import handle_diffusion_hooks
from .common import get_normalized_arr_from_diffusion
from .common import Txt2ImgDiffusionModel
from ..cv import WHModel
from ..schema import IImageNode
from ...core import Node
from ...core import Schema


class Txt2ImgSDModel(Txt2ImgDiffusionModel, WHModel):
    pass


@Node.register(sd_txt2img_name)
class SDTxt2ImgNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = Txt2ImgSDModel
        schema.description = "Use Stable Diffusion to generate image from text."
        return schema

    @classmethod
    async def warmup(cls) -> None:
        register_sd()

    async def execute(self) -> Dict[str, Image.Image]:
        data = Txt2ImgSDModel(**self.data)
        m = get_sd_from(APIs.SD, data)  # type: ignore
        size = data.w, data.h
        kwargs = handle_diffusion_model(m, data)
        await handle_diffusion_hooks(m, data, self, kwargs)
        res = m.txt2img(data.text, size=size, max_wh=data.max_wh, **kwargs)
        img_arr = res.numpy()[0]
        image = Image.fromarray(to_uint8(get_normalized_arr_from_diffusion(img_arr)))
        return {"image": image}


__all__ = [
    "Txt2ImgSDModel",
    "SDTxt2ImgNode",
]
