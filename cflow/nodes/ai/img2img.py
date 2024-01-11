import cv2

import numpy as np

from PIL import Image
from typing import Dict
from typing import Tuple
from typing import Optional
from pydantic import Field
from pydantic import BaseModel
from cftool.cv import to_rgb
from cftool.cv import to_uint8
from cftool.cv import ImageProcessor

from .common import APIs
from .common import TImage
from .common import sd_img2img_name
from .common import register_sd
from .common import get_sd_from
from .common import get_api_pool
from .common import register_esr
from .common import register_isnet
from .common import register_esr_anime
from .common import register_inpainting
from .common import handle_diffusion_model
from .common import handle_diffusion_hooks
from .common import get_image_from_diffusion_output
from .common import TranslatorAPI
from .common import ImageModel
from .common import DiffusionModel
from .common import Img2ImgModel
from .common import CallbackModel
from .common import KeepOriginalModel
from .common import Img2ImgDiffusionModel
from ..schema import IImageNode
from ...core import Node
from ...core import Schema


# img2img (stable diffusion)


class Img2ImgSDSettings(BaseModel):
    text: str = Field(..., description="The text that we want to handle.")
    wh: Tuple[int, int] = Field(
        (0, 0),
        description="The target output size, `0` means as-is",
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
class Img2ImgSDNode(IImageNode):
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
        diffusion_output = m.img2img(
            image,
            cond=[data.text],
            max_wh=data.max_wh,
            fidelity=data.fidelity,
            anchor=64,
            **kwargs,
        ).numpy()[0]
        image = get_image_from_diffusion_output(diffusion_output)
        return {"image": image}


# inpainting (LDM)


class Img2ImgInpaintingSettings(BaseModel):
    mask_url: TImage = Field(..., description="The inpainting mask.")
    refine_fidelity: Optional[float] = Field(
        None,
        description="""Refine fidelity used in inpainting.
> If not `None`, we will reference the original image with this fidelity to perform a refinement step.
""",
    )
    max_wh: int = Field(832, description="The maximum resolution.")


class Img2ImgInpaintingModel(
    DiffusionModel, KeepOriginalModel, Img2ImgInpaintingSettings, ImageModel
):
    pass


@Node.register("ai.img2img.inpainting")
class Img2ImgInpaintingNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = Img2ImgInpaintingModel
        schema.description = "Use LDM to perform image inpainting."
        return schema

    @classmethod
    async def warmup(cls) -> None:
        register_inpainting()

    async def execute(self) -> Dict[str, Image.Image]:
        data = Img2ImgInpaintingModel(**self.data)
        image = await self.get_image_from("url")
        mask = await self.get_image_from("mask_url")
        with get_api_pool().use(APIs.INPAINTING) as m:
            kwargs = handle_diffusion_model(m, data, always_uncond=False)
            await handle_diffusion_hooks(m, data, self, kwargs)
            mask_arr = np.array(mask)
            mask_arr[..., -1] = np.where(mask_arr[..., -1] > 0, 255, 0)
            mask = Image.fromarray(mask_arr)
            refine_fidelity = data.refine_fidelity
            diffusion_output = m.inpainting(
                image,
                mask,
                max_wh=data.max_wh,
                refine_fidelity=refine_fidelity,
                **kwargs,
            ).numpy()[0]
        output = get_image_from_diffusion_output(diffusion_output)
        if data.keep_original:
            fade = data.keep_original_num_fade_pixels
            output.putalpha(mask)
            if fade is None:
                output = Image.alpha_composite(image.convert("RGBA"), output)
            else:
                output = ImageProcessor.paste(output, image, num_fade_pixels=fade)
        return {"image": output}


# super resolution (Real-ESRGAN)


class Img2ImgSRSettings(BaseModel):
    is_anime: bool = Field(
        False,
        description="Whether the input image is an anime image or not.",
    )
    target_w: int = Field(0, description="The target width. 0 means as-is.")
    target_h: int = Field(0, description="The target height. 0 means as-is.")


class Img2ImgSRModel(CallbackModel, Img2ImgSRSettings, Img2ImgModel):
    max_wh: int = Field(832, description="The maximum resolution.")


def apply_sr(
    m: TranslatorAPI,
    image: Image.Image,
    max_wh: int,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    img_arr = m.sr(image, max_wh=max_wh).numpy()[0]
    img_arr = img_arr.transpose([1, 2, 0])
    h, w = img_arr.shape[:2]
    if target_w and target_h:
        larger = w * h < target_w * target_h
        img_arr = cv2.resize(
            img_arr,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4 if larger else cv2.INTER_AREA,
        )
    elif target_w or target_h:
        if target_w:
            k = target_w / w
            target_h = round(h * k)
        else:
            k = target_h / h
            target_w = round(w * k)
        img_arr = cv2.resize(
            img_arr,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
        )
    return to_uint8(img_arr)


@Node.register("ai.img2img.sr")
class Img2ImgSRNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = Img2ImgSRModel
        schema.description = "Super resolution."
        return schema

    @classmethod
    async def warmup(cls) -> None:
        register_esr()
        register_esr_anime()

    async def execute(self) -> Dict[str, Image.Image]:
        data = Img2ImgSRModel(**self.data)
        image = await self.get_image_from("url")
        api_key = APIs.ESR_ANIME if data.is_anime else APIs.ESR
        with get_api_pool().use(api_key) as m:
            uint8_image = apply_sr(m, image, data.max_wh, data.target_w, data.target_h)
        image = Image.fromarray(uint8_image)
        return {"image": image}


# salient object detection (isnet)


@Node.register("ai.img2img.sod")
class Img2ImgSODNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.description = "Salient object detection (cutout)."
        return schema

    @classmethod
    async def warmup(cls) -> None:
        register_isnet()

    async def execute(self) -> Dict[str, Image.Image]:
        image = await self.get_image_from("url")
        with get_api_pool().use(APIs.ISNET) as m:
            rgb = to_rgb(image)
            alpha = to_uint8(m.segment(rgb))
        image = Image.fromarray(alpha)
        return {"image": image}


__all__ = [
    "Img2ImgSDModel",
    "Img2ImgInpaintingModel",
    "Img2ImgSRModel",
    "Img2ImgSDNode",
    "Img2ImgInpaintingNode",
    "Img2ImgSRNode",
    "Img2ImgSODNode",
]
