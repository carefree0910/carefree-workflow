# OpenAI Nodes

"""
Please install with

`pip install carefree-workflow[openai]`

or

`pip install carefree-workflow[full]`

to use these nodes.
"""

from typing import Dict
from typing import Optional
from pydantic import Field
from pydantic import BaseModel

from .schema import TextModel
from .schema import ImageModel
from .schema import IWithOpenAINode
from ..core import Node
from ..core import Schema


class OpenAIImageToTextInput(ImageModel):
    prompt: Optional[str] = Field(None, description="Prompt for image captioning.")


@Node.register("openai.img2txt")
class OpenAIImageToTextNode(IWithOpenAINode):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            OpenAIImageToTextInput,
            TextModel,
            description="Image captioning with OpenAI API.",
        )

    async def execute(self) -> Dict[str, str]:
        image = await self.get_image_from("url")
        image_url = await self.openai_client.upload(image)
        prompt = self.data["prompt"]
        if prompt is None:
            prompt = "Describe this image to a person with impaired vision. Be short and concise."
        caption = await self.openai_client.client.chat.completions.create(
            model="gpt-4-vision-preview",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        )
        caption = caption.choices[0].message.content
        if not caption:
            raise RuntimeError("Empty or missing caption.")
        return {"text": caption}


class OpenAITextToImageInput(TextModel):
    size: str = Field("1024x1024", description="Image size.")
    model: str = Field("dall-e-3", description="Model name.")
    quality: str = Field("standard", description="Image quality.")


class OpenAITextToImageOutput(BaseModel):
    image_url: str = Field(..., description="The url of the generated image.")


@Node.register("openai.txt2img")
class OpenAITextToImageNode(IWithOpenAINode):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            OpenAITextToImageInput,
            OpenAITextToImageOutput,
            description="Image generation with OpenAI API.",
        )

    async def execute(self) -> Dict[str, str]:
        response = await self.openai_client.client.images.generate(
            model=self.data["model"],
            size=self.data["size"],
            quality=self.data["quality"],
            n=1,
            prompt=self.data["text"],
        )
        image_url = response.data[0].url
        if not image_url:
            raise RuntimeError("Empty or missing image url.")
        return {"image_url": image_url}


__all__ = [
    "OpenAIImageToTextNode",
    "OpenAITextToImageNode",
]
