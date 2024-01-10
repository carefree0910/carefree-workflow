"""
Stable Diffusion text to image + super resolution example.

This example shows how to perform `txt2img -> super resolution` workflow. Please install with:

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
    flow = (
        Flow()
        .push(SDTxt2ImgNode("txt2img", dict(w=w, h=h, text=text, seed=seed)))
        .push(Img2ImgSRNode("sr", injections=[Injection("txt2img", "image", "url")]))
    )
    render_workflow(flow).save("workflow.png")
    results = await flow.execute("sr", verbose=True)
    results["txt2img"]["image"].save("out.png")
    results["sr"]["image"].save("out_sr.png")


if __name__ == "__main__":
    asyncio.run(main())
