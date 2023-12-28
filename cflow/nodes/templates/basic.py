from cflow import *


class CustomInput(BaseModel):
    """The input data model"""


class CustomOutput(BaseModel):
    """The output data model"""


# The registered name should be unique and will be turned into the API endpoint, with the dots (`.`) being replaced by slashes (`/`).
# e.g. if the node is registered with `Node.register("foo.bar")`, the corresponding API endpoint will be `/foo/bar`.
@Node.register("<registered name>")
class CustomNode(Node):
    @classmethod
    def get_schema(cls):
        return Schema(
            CustomInput,
            CustomOutput,
            description="<description of the node>",
        )

    async def execute(self):
        """
        The implementation of the node, here are the resources you can get:

        - self.data: This will follow the schema of `CustomInput`.
        - self.http_session : A (shared) aiohttp.ClientSession instance.

        The results of this method should follow the schema of `CustomOutput`.
        """
