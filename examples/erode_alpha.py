"""
Erode alpha example.

This example shows how to erode the alpha channel of the given image.
It also uses `render_workflow` to render the workflow graph.
"""

import asyncio

from cflow import *


async def main() -> None:
    dog_rgba_url = "https://ailab-huawei-cdn.nolibox.com/aigc/images/af0de0afdc1445d29f104d6b38d296fe.png"
    flow = (
        Flow()
        .push(DownloadImageNode("dog", dict(url=dog_rgba_url)))
        .push(GetMaskNode("mask", injections=[Injection("dog", "image", "url")]))
        .push(
            ErodeNode(
                "erode",
                dict(kernel_size=13),
                [Injection("mask", "image", "url")],
            )
        )
        .push(
            SetAlphaNode(
                "merge",
                injections=[
                    Injection("dog", "image", "url"),
                    Injection("erode", "image", "mask_url"),
                ],
            )
        )
    )
    results = await flow.execute("merge", verbose=True)
    render_workflow(flow).save("workflow.png")
    results["dog"]["image"].save("dog.png")
    results["mask"]["image"].save("mask.png")
    results["erode"]["image"].save("mask_eroded.png")
    results["merge"]["image"].save("merged.png")


if __name__ == "__main__":
    asyncio.run(main())
