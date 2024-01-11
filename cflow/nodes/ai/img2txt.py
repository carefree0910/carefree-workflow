from PIL import Image
from typing import Dict
from cftool.cv import to_rgb
from cftool.cv import restrict_wh

from .common import APIs
from .common import get_api_pool
from .common import register_blip
from .common import TextModel
from .common import ImageModel
from .common import MaxWHModel
from ..schema import IWithImageNode
from ...core import Node
from ...core import Schema


class Img2TxtInput(MaxWHModel, ImageModel):
    pass


@Node.register("ai.img2txt.caption")
class Img2TxtCaptionNode(IWithImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(Img2TxtInput, TextModel, description="Image captioning.")

    @classmethod
    async def warmup(cls) -> None:
        register_blip()

    async def execute(self) -> Dict[str, Image.Image]:
        data = Img2TxtInput(**self.data)
        image = await self.get_image_from("url")
        w, h = image.size
        w, h = restrict_wh(w, h, data.max_wh)
        image = image.resize((w, h), resample=Image.LANCZOS)
        image = to_rgb(image)
        with get_api_pool().use(APIs.BLIP) as m:
            caption = m.caption(image)
        return {"text": caption}


__all__ = [
    "Img2TxtInput",
    "Img2TxtCaptionNode",
]
