"""
Stable Diffusion image to image example.

This example shows how to use `SDImg2ImgNode` node to perform image to image. Please install with:

`pip install carefree-workflow[ai]`

or

`pip install carefree-workflow[full]`

to run this example.
"""

import asyncio

from cflow import *


async def main() -> None:
    wh = (640, 425)
    url = "https://ailab-huawei-cdn.nolibox.com/upload/images/ba4a27c434394bf684890643890970d2.png"
    text = "A lovely little cat."
    seed = 123
    data = dict(wh=wh, url=url, text=text, seed=seed)
    flow = Flow().push(Img2ImgSDNode("img2img", data))
    results = await flow.execute("img2img", verbose=True)
    results["img2img"]["image"].save("out.png")


if __name__ == "__main__":
    asyncio.run(main())
