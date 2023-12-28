from cflow import *


class CustomInput(BaseModel):
    """The input data model"""


class CustomAPIOutput(BaseModel):
    """
    The api output data model.

    This is useful when your 'raw' results of the node is complex (e.g., contains `PIL.Image`),
    and you still want to serve the node independently.

    In this case, you need to implement the `get_api_response` method to do conversions.
    """


# The registered name should be unique and will be turned into the API endpoint, with the dots (`.`) being replaced by slashes (`/`).
# e.g. if the node is registered with `Node.register("foo.bar")`, the corresponding API endpoint will be `/foo/bar`.
@Node.register("<registered name>")
class CustomNode(Node):
    @classmethod
    def get_schema(cls):
        return Schema(
            CustomInput,
            api_output_model=CustomAPIOutput,
            output_names=["<keys of the 'raw' results>"],
            description="<description of the node>",
        )

    @classmethod
    async def get_api_response(cls, results):
        """
        The conversion of the 'raw' results.

        This method will be called when the node is executed through the API.
        The results of this method should follow the schema of `CustomAPIOutput`.

        > One common scenario is to convert the `PIL.Image`s to base64 strings.
        > You can easily achieve this by utilizing the `cftool` library:

        ```python
        from cftool.cv import to_base64

        base_64 = to_base64(image)
        ```
        """

    async def execute(self):
        """
        The implementation of the node, here are the resources you can get:

        - self.data: This will follow the schema of `CustomInput`.
        - self.http_session : A (shared) aiohttp.ClientSession instance.

        The results of this method should be a `dict` and has the same keys as `output_names`.
        """
