"""
Hello world example.

This example implements a `HelloNode` which takes a `name` as input and returns the `name` and a `greeting` as outputs.
Then, the `EchoNode` is used to print out the results from `HelloNode`.
This example also uses `render_workflow` to render the workflow graph, and `dump` to dump the workflow to a JSON file.
It also shows how to load a workflow from a JSON file.
"""

import asyncio

from cflow import *


class HelloInput(BaseModel):
    name: str


class HelloOutput(BaseModel):
    name: str
    greeting: str


@Node.register("hello")
class HelloNode(Node):
    @classmethod
    def get_schema(cls):
        return Schema(HelloInput, HelloOutput)

    async def execute(self):
        name = self.data["name"]
        return {"name": name, "greeting": f"Hello, {name}!"}


async def main() -> None:
    flow = (
        Flow()
        .push(HelloNode("A", dict(name="foo")))
        .push(HelloNode("B", dict(name="bar")))
        .push(
            EchoNode(
                "Echo",
                dict(messages=[None, None, "Hello, World!"]),
                injections=[
                    Injection("A", "name", "messages.0"),
                    Injection("B", "greeting", "messages.1"),
                ],
            )
        )
    )
    results = await flow.execute("Echo")
    render_workflow(flow).save("workflow.png")
    flow.dump("workflow.json")
    # Should print something like: {'A': {'name': 'foo', ...}, 'B': {...}, 'Echo': {...}}
    print("> Results:", dict(results))


async def load(path: str = "workflow.json") -> None:
    flow = Flow.load(path)
    results = await flow.execute("Echo")
    render_workflow(flow).save("loaded.png")
    flow.dump("loaded.json")
    print("> Results:", dict(results))


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(load())
