"""
Stable Diffusion text to image example.

This example shows how to use `SDTxt2ImgNode` node to generate an image of the given text. Please install with:

`pip install carefree-workflow[ai]`

or

`pip install carefree-workflow[full]`

to run this example.
"""

import asyncio

from cflow import *


async def main() -> None:
    w = 512
    h = 512
    text = "A lovely little cat."
    seed = 123
    flow = Flow().push(Txt2ImgSDNode("txt2img", dict(w=w, h=h, text=text, seed=seed)))
    results = await flow.execute("txt2img", verbose=True)
    results["txt2img"]["image"].save("out.png")


if __name__ == "__main__":
    asyncio.run(main())
