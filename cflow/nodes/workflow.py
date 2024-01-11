# Workflow as Node

from PIL import Image
from typing import Dict
from pydantic import Field
from pydantic import BaseModel
from cftool.misc import shallow_copy_dict

from .cv import ErodeNode
from .cv import ErodeInput
from .cv import GetMaskNode
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


__all__ = [
    "ErodeAlphaNode",
]
