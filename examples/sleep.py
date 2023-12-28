import time
import asyncio

from cflow import *


class SleepInput(BaseModel):
    time: int


class SleepOutput(BaseModel):
    message: str


@Node.register("sleep")
class SleepNode(Node):
    @classmethod
    def get_schema(cls):
        return Schema(SleepInput, SleepOutput)

    async def execute(self):
        t = self.data["time"]
        time.sleep(t)
        return {"message": f"[{self.key}] Slept for {t} seconds."}


@Node.register("async_sleep")
class AsyncSleepNode(Node):
    async def execute(self):
        t = self.data["time"]
        await asyncio.sleep(t)
        return {"message": f"[{self.key}] Slept for {t} seconds."}


async def main() -> None:
    get_injection = lambda key: Injection(key, "message", "messages.0")
    flow = (
        Flow()
        # by setting `offload=True`, even 'sync' nodes can be executed asynchronously
        # this means `A` & `B` will be executed concurrently
        # if not specified, `B` will be executed only after `A` is finished
        .push(SleepNode("A", dict(time=1), offload=True))
        # by specifying the same `lock_key` between `B` & `C`,
        # `C` will be executed only after `B` is finished because it is 'locked'
        .push(AsyncSleepNode("B", dict(time=2), lock_key="$"))
        .push(AsyncSleepNode("C", dict(time=3), lock_key="$"))
        .push(EchoNode("Echo A", dict(messages=[]), [get_injection("A")]))
        .push(EchoNode("Echo B", dict(messages=[]), [get_injection("B")]))
        .push(EchoNode("Echo C", dict(messages=[]), [get_injection("C")]))
    )
    # gather `Echo A`, `Echo B`, `Echo C` to 'wait' for them to finish
    gathered = flow.gather("Echo A", "Echo B", "Echo C")
    # setting `verbose=True` will print out debug logs,
    # which can show the execution order of the nodes more clearly
    await flow.execute(gathered, verbose=True)
    render_workflow(flow).save("workflow.png")
    flow.dump("workflow.json")


if __name__ == "__main__":
    asyncio.run(main())
