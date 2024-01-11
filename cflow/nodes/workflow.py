# Workflow as Node

from PIL import Image
from typing import Dict
from pydantic import Field
from pydantic import BaseModel
from cftool.misc import shallow_copy_dict

from .ai import Img2ImgInpaintingNode
from .ai import Img2ImgInpaintingInput
from .cv import FadeNode
from .cv import BlurNode
from .cv import ErodeNode
from .cv import DilateNode
from .cv import ErodeInput
from .cv import GetMaskNode
from .cv import BinarizeNode
from .cv import SetAlphaNode
from .schema import IImageNode
from .common import DownloadImageNode
from ..core import Node
from ..core import Flow
from ..core import Schema
from ..core import Injection


class VerboseModel(BaseModel):
    verbose: bool = Field(False, description="Whether to print debug logs.")


class ErodeAlphaInput(VerboseModel, ErodeInput):
    pass


@Node.register("workflow.erode_alpha")
class ErodeAlphaNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = ErodeAlphaInput
        schema.description = "Erode the alpha channel of an image."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        data = shallow_copy_dict(self.data)
        url = data.pop("url")
        verbose = data.pop("verbose", False)
        flow = (
            Flow()
            .push(DownloadImageNode("input", dict(url=url)))
            .push(GetMaskNode("mask", injections=[Injection("input", "image", "url")]))
            .push(ErodeNode("erode", data, [Injection("mask", "image", "url")]))
            .push(
                SetAlphaNode(
                    "merge",
                    injections=[
                        Injection("input", "image", "url"),
                        Injection("erode", "image", "mask_url"),
                    ],
                )
            )
        )
        results = await flow.execute("merge", verbose=verbose)
        return results["merge"]


class AdvancedInpaintingInput(VerboseModel, Img2ImgInpaintingInput):
    binarize_threshold: int = Field(16, description="Threshold for binarization.")


@Node.register("workflow.advanced_inpainting")
class AdvancedInpaintingNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = Img2ImgInpaintingInput
        schema.description = "Use LDM & tricks to perform advanced image inpainting."
        return schema

    @classmethod
    async def warmup(cls) -> None:
        await Img2ImgInpaintingNode.warmup()

    async def execute(self) -> Dict[str, Image.Image]:
        data = shallow_copy_dict(self.data)
        url = data.pop("url")
        mask_url = data.pop("mask_url")
        binarize_threshold = data.pop("binarize_threshold", 16)
        verbose = data.pop("verbose", False)
        flow = (
            Flow()
            .push(DownloadImageNode("image", dict(url=url)))
            .push(DownloadImageNode("mask", dict(url=mask_url)))
            .push(
                BinarizeNode(
                    "binarize",
                    dict(threshold=binarize_threshold),
                    injections=[Injection("mask", "image", "url")],
                )
            )
            .push(
                DilateNode(
                    "dilate",
                    dict(kernel_size=7),
                    [Injection("binarize", "image", "url")],
                )
            )
            .push(BlurNode("blur", injections=[Injection("dilate", "image", "url")]))
            .push(
                Img2ImgInpaintingNode(
                    "inpainting",
                    injections=[
                        Injection("image", "image", "url"),
                        Injection("blur", "image", "mask_url"),
                    ],
                )
            )
            .push(BlurNode("blur_blur", injections=[Injection("blur", "image", "url")]))
            .push(
                SetAlphaNode(
                    "set_alpha",
                    injections=[
                        Injection("inpainting", "image", "url"),
                        Injection("blur_blur", "image", "mask_url"),
                    ],
                )
            )
            .push(
                FadeNode(
                    "fade",
                    dict(force_rgb=True),
                    [
                        Injection("set_alpha", "image", "url"),
                        Injection("image", "image", "bg_url"),
                    ],
                )
            )
        )
        results = await flow.execute("fade", verbose=verbose)
        return results["fade"]


__all__ = [
    "ErodeAlphaInput",
    "AdvancedInpaintingInput",
    "ErodeAlphaNode",
    "AdvancedInpaintingNode",
]
