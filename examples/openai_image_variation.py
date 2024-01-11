"""
Image variation example.

This example shows how to use `OpenAI` nodes to generate a variation of the given image. Please:
> - set `OPENAI_API_KEY` environment variable
> - install with `pip install carefree-workflow[openai]` or `pip install carefree-workflow[full]`

to run this example.
"""

import asyncio

from cflow import *


async def main() -> None:
    cat_url = "https://ailab-huawei-cdn.nolibox.com/upload/images/ba4a27c434394bf684890643890970d2.png"
    injections = [Injection("img2txt", "text", "text")]
    flow = (
        Flow()
        .push(OpenAIImageToTextNode("img2txt", dict(url=cat_url)))
        .push(OpenAITextToImageNode("txt2img", injections=injections))
    )
    results = await flow.execute("txt2img", verbose=True)
    render_workflow(flow).save("workflow.png")
    print("> generated image url:", results["txt2img"]["image_url"])


if __name__ == "__main__":
    asyncio.run(main())
