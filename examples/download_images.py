"""
Download images example.

This example shows how to use `DownloadImageNode` to download multiple images concurrently.
It also shows how to use `gather` to wait for multiple nodes to finish, and how the results are organized.
It also uses `render_workflow` to render the workflow graph, and `dump` to dump the workflow to a JSON file.
"""

import asyncio

from cflow import *


async def main() -> None:
    cat_url = "https://ailab-huawei-cdn.nolibox.com/upload/images/ba4a27c434394bf684890643890970d2.png"
    dog_url = "https://ailab-huawei-cdn.nolibox.com/upload/images/4814c36b452f47268ba77d54cc706f88.png"
    flow = (
        Flow()
        .push(DownloadImageNode("cat", dict(url=cat_url)))
        .push(DownloadImageNode("dog", dict(url=dog_url)))
    )
    gathered = flow.gather("cat", "dog")
    results = await flow.execute(gathered, verbose=True)
    render_workflow(flow).save("workflow.png")
    results[gathered]["cat"]["image"].save("cat.png")
    results[gathered]["dog"]["image"].save("dog.png")


if __name__ == "__main__":
    asyncio.run(main())
