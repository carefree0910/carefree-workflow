"""
Auto foreground removal example.

This example shows how to perform `cutout -> (advanced) inpainting` workflow. Please install with:

`pip install carefree-workflow[ai]`

or

`pip install carefree-workflow[full]`

to run this example.
"""

import asyncio

from cflow import *


async def main() -> None:
    url = "https://ailab-huawei-cdn.nolibox.com/upload/images/45d3b020ff544520b23f4aa4f149b0d9.png"
    image_injection = Injection("download", "image", "url")
    flow = (
        Flow()
        .push(DownloadImageNode("download", dict(url=url)))
        .push(Img2ImgSODNode("sod", injections=[image_injection]))
        .push(
            AdvancedInpaintingNode(
                "inpainting",
                injections=[image_injection, Injection("sod", "image", "mask_url")],
            )
        )
    )
    render_workflow(flow, layout="circular_layout").save("workflow.png")
    results = await flow.execute("inpainting", verbose=True)
    results["download"]["image"].save("original.png")
    results["inpainting"]["image"].save("out.png")


if __name__ == "__main__":
    asyncio.run(main())
