"""
Download images example.

This example shows how to use `DownloadImageNode` to download multiple images concurrently.
It also shows how to use `gather` to wait for multiple nodes to finish, and how the results are organized.
It also uses `render_workflow` to render the workflow graph, and `dump` to dump the workflow to a JSON file.
"""

import asyncio

from cflow import *


async def main() -> None:
    cat_url = "https://cdn.pixabay.com/photo/2016/01/20/13/05/cat-1151519_1280.jpg"
    dog_url = "https://cdn.pixabay.com/photo/2020/03/31/19/20/dog-4988985_1280.jpg"
    flow = (
        Flow()
        .push(DownloadImageNode("cat", dict(url=cat_url)))
        .push(DownloadImageNode("dog", dict(url=dog_url)))
    )
    gathered = flow.gather("cat", "dog")
    results = await flow.execute(gathered, verbose=True)
    render_workflow(flow).save("workflow.png")
    results[gathered]["cat"]["image"].save("cat.jpg")
    results[gathered]["dog"]["image"].save("dog.jpg")


if __name__ == "__main__":
    asyncio.run(main())
