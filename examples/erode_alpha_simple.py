"""
Erode alpha example.

This example shows how to use `ErodeAlphaNode` to erode the alpha channel of the given image.
It also uses `render_workflow` to render the workflow graph.
"""

import asyncio

from cflow import *


async def main() -> None:
    dog_rgba_url = "https://ailab-huawei-cdn.nolibox.com/aigc/images/af0de0afdc1445d29f104d6b38d296fe.png"
    flow = Flow().push(ErodeAlphaNode("erode", dict(url=dog_rgba_url, kernel_size=13)))
    results = await flow.execute("erode", verbose=True)
    render_workflow(flow).save("workflow.png")
    results["erode"]["image"].save("eroded.png")


if __name__ == "__main__":
    asyncio.run(main())
