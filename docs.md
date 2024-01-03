# `carefree-workflow` Documentation

Here are some design principles of `carefree-workflow`:
- `workflow` is a `DAG` (directed acyclic graph) of `nodes`.
- `workflow` is actually constructed by a set of `nodes` with `injections` defined.
  - `injections` indicate the dependencies between `nodes`.
- Every `node`, as well as the `workflow` itself, can be used in both `functional` and `API` ways.
- Every `node` should take `dict` as inputs and return `dict` as outputs.

And below will be the detailed documents of:
- Installation.
- The general introduction of `node` and `workflow`.
- All the nodes supported in `carefree-workflow`.

We'll also include some examples at the end to help you understand how to use `carefree-workflow` in your projects.

# Installation

`carefree-workflow` requires Python 3.8 or higher.

```bash
pip install carefree-workflow
```

or

```bash
git clone https://github.com/carefree0910/carefree-workflow.git
cd carefree-workflow
pip install -e .
```

# Node

Every `node` in `carefree-workflow` should inherit from `cflow.Node`:

```python
@dataclass
class Node(ISerializableDataClass, metaclass=ABCMeta):
    """
    A Node class that represents a node in a workflow.

    This class is abstract and should be subclassed.

    Attributes
    ----------
    key : str, optional
        The key of the node, should be unique with respect to the workflow.
    data : Any, optional
        The data associated with the node.
    injections : List[Injection], optional
        A list of injections of the node.
    offload : bool, optional
        A flag indicating whether the node should be offloaded.
    lock_key : str, optional
        The lock key of the node.
    executing : bool, optional
        A runtime attribute indicating whether the node is currently executing.

    Methods
    -------
    async execute() -> Any
        Abstract method that should return the results.

    @classmethod
    get_schema() -> Optional[Schema]
        Optional method that returns the schema of the node.
        Implement this method can help us auto-generate UIs, APIs and documents.
    @classmethod
    async get_api_response(results: Dict[str, Any]) -> Any
        Optional method that returns the API response of the node from its 'raw' results.
        Implement this method to handle complex API responses (e.g. 'PIL.Image').
    async initialize(flow: Flow) -> None
        Optional method that will be called before the execution of the workflow.
        Implement this method to do heavy initializations (e.g. loading AI models).

    """

    key: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    injections: List[Injection] = field(default_factory=list)
    offload: bool = False
    lock_key: Optional[str] = None
    # runtime attribute, should not be touched and will not be serialized
    executing: bool = False
    shared_pool: Dict[str, Any] = field(default_factory=dict)
```

It looks complicated, but `node` can actually be simply understood as a `function`, except:
- It can be used in an `API` way **automatically**, as long as it implements the `get_schema` method.
- It can be used in a `workflow`, which means:
  - Its input(s) can be the output(s) from other `node`.
  - Its output(s) can be the input(s) of other `node`.

The second feature is achieved by `injections`, which is represented by:
- `Injection`, if used in a `functional` way.
- `InjectionModel`, if used in an `API` way.

`InjectionModel` will be introduced later ([API usage](#api-usage)), and here is the definition of `Injection`:

```python
@dataclass
class Injection:
    """
    A dataclass that represents an injection to the current node.

    Attributes
    ----------
    src_key : str
        The key of the dependent node.
    src_hierarchy : str | None
        The 'src_hierarchy' of the dependent node's results that the current node depends on.
        - 'src_hierarchy' can be very complex:
          - use 'int' as 'list' index, and 'str' as 'dict' key.
          - use '.' to represent nested structure.
          - for example, you can use 'a.0.b' to indicate 'results["a"][0]["b"]'.
        - If 'None', all results of the dependent node will be used.
    dst_hierarchy : str
        The 'dst_hierarchy' of the current node's 'data'.
        - 'dst_hierarchy' can be very complex:
          - use 'int' as 'list' index, and 'str' as 'dict' key.
          - use '.' to represent nested structure.
          - for example, if you want to inject to 'data["a"][0]["b"]', you can use 'a.0.b'
          as the 'dst_hierarchy'.

    """

    src_key: str
    src_hierarchy: Optional[str]
    dst_hierarchy: str
```

> Example of how to use `Injection` will also be introduced later ([Functional usage](#functional-usage)).

## Example

Here's an example of how to define a custom `node`:

```python
@Node.register("hello")
class HelloNode(Node):
    async def execute(self):
        name = self.data["name"]
        return {"name": name, "greeting": f"Hello, {name}!"}
```

In the above example, we defined a `node` named `hello`, which takes a `name` as input and returns the `name` itself and a `greeting` as outputs.

To make it 'collectable' by the automated system, we can implement the `get_schema` method:

```python
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
```

This will help us automatically generate the API endpoint as well as the documentation.

# Workflow

```python
class Flow(Bundle[Node]):
    """
    A Flow class that represents a workflow.

    Attributes
    ----------
    edges : Dict[str, List[Edge]]
        The dependencies of the workflow.
        - The key is the destination node key.
        - The value is a list of edges that indicates the dependencies
          of the destination node.
    latest_latencies : Dict[str, Dict[str, float]]
        The latest latencies of the workflow.

    Methods
    -------
    push(node: Node) -> Flow:
        Pushes a node into the workflow.
    gather(*targets: str) -> str:
        Gathers targets into a single node, and returns the key of the node.
    to_json() -> Dict[str, Any]:
        Converts the workflow to a JSON object.
    from_json(cls, data: Dict[str, Any]) -> Flow:
        Creates a workflow from a JSON object.
    dump(path: TPath) -> None:
        Dumps the workflow to a (JSON) file.
    load(cls, path: TPath) -> Flow:
        Loads a workflow from a (JSON) file.
    get_reachable(target: str) -> Set[str]:
        Gets the reachable nodes from a target.
    run(...) -> None:
        Runs a single node in the workflow.
    execute(...) -> Dict[str, Any]:
        Executes the entire workflow.

    """
```

The key method used by users will be the `execute` method, which is defined as:

```python
async def execute(
    self,
    target: str,
    intermediate: Optional[List[str]] = None,
    *,
    return_api_response: bool = False,
    return_if_exception: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Executes the workflow ending at the 'target' node.

    Parameters
    ----------
    target : str
        The key of the target node which the execution will end at.
    intermediate : List[str], optional
        A list of intermediate nodes that will be returned. Default is 'None'.
        - Only useful when 'return_api_response' is 'True'.
        - If 'None', no intermediate nodes will be returned.
    return_if_exception : bool, optional
        If 'True', the function will return even if an exception occurs. Default is 'False'.
    return_api_response : bool, optional
        If 'True', the function will:
        - Only return the results of the 'target' node & the 'intermediate' nodes.
        - Call 'get_api_response' on the results to get the final API response.
    verbose : bool, optional
        If 'True', the function will print detailed logs. Default is 'False'.

    Returns
    -------
    dict
        A dictionary containing the results of the execution.
        - If 'return_api_response' is 'True', only outputs of the 'target' node can be accessed
        (via 'results[target]').
        - Otherwise, outputs of all nodes can be accessed (via 'results[key]', where 'key' is
        the key of the node).
        - If an exception occurs during the execution, the dictionary will contain
        a key 'EXCEPTION_MESSAGE_KEY' with the error message as the value.

    """
```

## Functional usage

A typical procedure of using `workflow` in a `functional` way is as follows:
- Define your custom `nodes` by inheriting from `cflow.Node` (if needed).
- Define your `workflow` by using `cflow.Flow` in a chainable way (`cflow.Flow().push(...).push(...)`).
  - Use `flow.gather(...)` if you have multiple targets.
- Call `await workflow.execute(...)` to execute the `workflow` with the given inputs.

Here is a simple example:

```python
import asyncio

from cflow import *

@Node.register("hello")
class HelloNode(Node):
    async def execute(self):
        name = self.data["name"]
        return {"name": name, "greeting": f"Hello, {name}!"}

async def main():
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
    await flow.execute("Echo")

if __name__ == "__main__":
    asyncio.run(main())
```

Running the above codes will yield something like:

```text
[17:30:27] foo
           Hello, bar!
           Hello, World!
```

> More examples can be found at the end of this document ([Coding Examples](#coding-examples)).

## API usage

Here are some important input data models when you want to use `workflow` in an `API` way.

> Examples can be found at the end of this document ([Workflow JSON Examples](#workflow-json-examples)).

### `InjectionModel`

```python
class SrcKey(BaseModel):
    src_key: str = Field(..., description="The key of the dependent node.")
```

```python
class LoopBackInjection(BaseModel):
    """
    A dataclass that represents a loop back injection to the current node.

    > This is the same as 'Injection', except the 'src_key' will always be the
    key of the previous node in the loop.
    """

    src_hierarchy: Optional[str] = Field(
        ...,
        description="""The 'src_hierarchy' of the dependent node's results that the current node depends on.
- 'src_hierarchy' can be very complex:
  - use 'int' as 'list' index, and 'str' as 'dict' key.
  - use '.' to represent nested structure.
  - for example, you can use 'a.0.b' to indicate 'results["a"][0]["b"]'.
- If 'None', all results of the dependent node will be used.""",
    )
    dst_hierarchy: str = Field(
        ...,
        description="""The 'dst_hierarchy' of the current node's 'data'.
- 'dst_hierarchy' can be very complex:
  - use 'int' as 'list' index, and 'str' as 'dict' key.
  - use '.' to represent nested structure.
  - for example, if you want to inject to 'data["a"][0]["b"]', you can use 'a.0.b'
  as the 'dst_hierarchy'.""",
    )
```

```python
class InjectionModel(LoopBackInjection, SrcKey):
    pass
```

### `NodeModel`

```python
class NodeModel(BaseModel):
    key: str = Field(
        ...,
        description="The key of the node, should be unique with respect to the workflow.",
    )
    type: str = Field(
        ...,
        description="The type of the node, should be the one when registered.",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="The data associated with the node.",
    )
    injections: List[InjectionModel] = Field(
        default_factory=list,
        description="A list of injections of the node.",
    )
    offload: bool = Field(
        False,
        description="A flag indicating whether the node should be offloaded.",
    )
    lock_key: Optional[str] = Field(None, description="The lock key of the node.")
```

### `WorkflowModel`

```python
class WorkflowModel(BaseModel):
    target: str = Field(..., description="The target output node of the workflow.")
    intermediate: Optional[List[str]] = Field(
        None,
        description="The intermediate nodes that you want to get outputs from.",
    )
    nodes: List[NodeModel] = Field(..., description="A list of nodes in the workflow.")
    return_if_exception: bool = Field(
        False,
        description="Whether to return partial results if exception occurs.",
    )
    verbose: bool = Field(False, description="Whether to print debug logs.")
```

# Schema

## Common Enums

### `ResizeMode`

```python
class ResizeMode(str, DocEnum):
    FILL = "fill"
    FIT = "fit"
    COVER = "cover"
```

### `Resampling`

```python
class Resampling(str, DocEnum):
    NEAREST = "nearest"
    BOX = "box"
    BILINEAR = "bilinear"
    HAMMING = "hamming"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
```

## Common Data Models

### `TextModel`

```python
class TextModel(DocModel):
    text: str = Field(..., description="The text.")
```

### `ImageModel`

```python
class ImageModel(DocModel):
    url: TImage = Field(..., description="The url / PIL.Image instance of the image.")
```

### `ImageAPIOuput`

```python
class ImageAPIOuput(DocModel):
    image: str = Field(..., description="The base64 encoded image.")
```

### `WHModel`

```python
class WHModel(DocModel):
    w: int = Field(..., description="Width of the output image.")
    h: int = Field(..., description="Height of the output image")
```

### `LTRBModel`

```python
class LTRBModel(DocModel):
    lt_rb: Tuple[int, int, int, int] = Field(
        ...,
        description="The left-top and right-bottom points.",
    )
```

### `ResamplingModel`

```python
class ResamplingModel(DocModel):
    resampling: Resampling = Field(Resampling.BILINEAR, description="The resampling.")
```

### `BaseAffineModel`

```python
class BaseAffineModel(DocModel):
    a: float = Field(..., description="'a' of the affine matrix")
    b: float = Field(..., description="'b' of the affine matrix")
    c: float = Field(..., description="'c' of the affine matrix")
    d: float = Field(..., description="'d' of the affine matrix")
    e: float = Field(..., description="'e' of the affine matrix")
    f: float = Field(..., description="'f' of the affine matrix")
    wh_limit: int = Field(
        16384,
        description="maximum width or height of the output image",
    )
```

# Supported Nodes

## LoopNode

### Description

A node that represents a loop of another node.

### Inputs

```python
class LoopInput(BaseModel):
    base_node: str = Field(..., description="The node to be looped.")
    base_data: Dict[str, Any] = Field(default_factory=dict, description="Base data.")
    loop_values: Dict[str, List[Any]] = Field(
        ...,
        description="""The values to be looped.
> - The keys should be the 'target hierarchy' of the 'data'
> - The values should be a list of values to be looped & injectedinto the 'target hierarchy'.
> - All values should have the same length.

For example, if you want to loop 'data["a"]' with values '[1, 2]', and loop 'data["b"][0]["c"]' with values '[3, 4]', you can use:
'''python
{
    "a": [1, 2],
    "b.0.c": [3, 4],
}
'''
""",
    )
    loop_back_injections: Optional[List[LoopBackInjection]] = Field(
        None,
        description="The loop back injections.\n"
        "> - If this is set, the results from the previous step in the loop will be "
        "injected into the current node's 'data'.\n"
        "> - If 'None', no injection will be performed, and all nodes will be "
        "executed in parallel.",
    )
    extract_hierarchy: Optional[str] = Field(
        None,
        description="The hierarchy of the results to be extracted.\n"
        "> - If 'None', all results will be preserved.",
    )
    verbose: bool = Field(False, description="Whether to print debug logs.")
```

### Functional Outputs

```python
class LoopOutput(BaseModel):
    results: List[Any] = Field(..., description="The results of the loop.")
```

### API Outputs

*Same as the functional outputs.*

### Source Codes

```python
@Node.register("common.loop")
class LoopNode(Node):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            LoopInput,
            LoopOutput,
            description="A node that represents a loop of another node.",
        )

    async def execute(self) -> Dict[str, List[Dict[str, Any]]]:
        t_node = Node.get(self.data["base_node"])
        if t_node is None:
            raise ValueError(f"node `{self.data['base_node']}` is not defined")
        base_data = self.data["base_data"]
        loop_values = self.data["loop_values"]
        loop_back_injections = self.data["loop_back_injections"]
        loop_keys = list(loop_values)
        lengths = [len(loop_values[k]) for k in loop_keys]
        if len(set(lengths)) != 1:
            raise ValueError(
                "all loop values should have the same length, "
                f"but lengths are {lengths}"
            )
        n = lengths[0]
        flow = Flow()
        for i in range(n):
            i_data = shallow_copy_dict(base_data)
            for k in loop_keys:
                i_data[k] = loop_values[k][i]
            if loop_back_injections is None:
                i_injections = []
            else:
                i_injections = list(map(shallow_copy_dict, loop_back_injections))
                i_injections = [Injection(**d) for d in i_injections]
            flow.push(t_node(str(i), i_data, i_injections))
        target = flow.gather(*map(str, range(n)))
        results = await flow.execute(target, verbose=self.data["verbose"])
        extracted = [results[str(i)] for i in range(n)]
        extract_hierarchy = self.data["extract_hierarchy"]
        if extract_hierarchy is not None:
            extracted = [extract_from(rs, extract_hierarchy) for rs in extracted]
        return {"results": extracted}
```

## GatherNode

### Description

A node that is used to gather other nodes' results.
> - This is useful when you have multiple targets to collect results from.
> - If you are programming in Python, you can use `flow.gather` to make things easier.

### Inputs

*Undefined*

### Functional Outputs

*Undefined*

### API Outputs

*Same as the functional outputs.*

### Source Codes

```python
@Node.register(GATHER_NODE)
class GatherNode(Node):
    flow: Optional[Flow] = None

    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            description="A node that is used to gather other nodes' results.\n"
            "> - This is useful when you have multiple targets to collect results from.\n"
            "> - If you are programming in Python, you can use `flow.gather` to make things easier.",
        )

    @classmethod
    async def get_api_response(cls, results: Dict[str, Any]) -> Dict[str, Any]:
        if cls.flow is None:
            console.warn(
                "`flow` is not provided for `GatherNode`, raw results will be returned "
                "and `get_api_response` might not work as expected"
            )
            return results
        keys = list(results)
        node_names = [cls.flow.get(k).data.__identifier__ for k in keys]
        t_nodes = list(map(Node.get, node_names))
        tasks = [t.get_api_response(results[k]) for k, t in zip(keys, t_nodes)]
        converted = await asyncio.gather(*tasks)
        return {k: v for k, v in zip(keys, converted)}

    async def initialize(self, flow: Flow) -> None:
        await super().initialize(flow)
        self.flow = flow

    async def execute(self) -> Dict[str, Any]:
        return self.data

    def from_info(self, info: Dict[str, Any]) -> None:
        super().from_info(info)
        for injection in self.injections:
            if injection.src_hierarchy is not None:
                raise ValueError(
                    "`GatherNode` should always use `src_hierarchy=None` "
                    f"for injections, but `{injection}` is found"
                )
            if injection.src_key != injection.dst_hierarchy:
                raise ValueError(
                    "`GatherNode` should always use `src_key=dst_hierarchy` "
                    f"for injections, but `{injection}` is found"
                )
```

## ParametersNode

### Description

Setup parameters.
> - This is often used in a pre-defined workflow JSON to decide which parameters to be exposed to the user.
> - See [examples](https://github.com/carefree0910/carefree-workflow/tree/main/examples/workflows) for reference.

### Inputs

```python
class ParametersModel(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict, description="The parameters.")
```

### Functional Outputs

```python
class ParametersModel(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict, description="The parameters.")
```

### API Outputs

*Same as the functional outputs.*

### Source Codes

```python
@Node.register("common.parameters")
class ParametersNode(Node):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            ParametersModel,
            ParametersModel,
            description="Setup parameters.\n"
            "> - This is often used in a pre-defined workflow JSON to decide "
            "which parameters to be exposed to the user.\n"
            "> - See [examples](https://github.com/carefree0910/carefree-workflow/tree/main/examples/workflows) for reference.",
        )

    async def execute(self) -> Dict[str, Any]:
        return self.data
```

## EchoNode

### Description

Echo the given message(s).

### Inputs

```python
class EchoModel(BaseModel):
    messages: Union[str, List[str]]
```

### Functional Outputs

```python
class EchoModel(BaseModel):
    messages: Union[str, List[str]]
```

### API Outputs

*Same as the functional outputs.*

### Source Codes

```python
@Node.register("common.echo")
class EchoNode(Node):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(EchoModel, EchoModel, description="Echo the given message(s).")

    async def execute(self) -> Dict[str, Union[str, List[str]]]:
        messages = self.data["messages"]
        if isinstance(messages, str):
            messages = [messages]
        for message in messages:
            log(message)
        return self.data
```

## DownloadImageNode

### Description

Download an image from the given url.

### Inputs

```python
class ImageModel(DocModel):
    url: TImage = Field(..., description="The url / PIL.Image instance of the image.")
```

### Functional Outputs

'DownloadImageNode' has following outputs:
- image

### API Outputs

```python
class ImageAPIOuput(DocModel):
    image: str = Field(..., description="The base64 encoded image.")
```

### Source Codes

```python
@dataclass
@Node.register("common.download_image")
class DownloadImageNode(IImageNode):
    offload: bool = True

    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.description = "Download an image from the given url."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        image = await self.get_image_from("url")
        return {"image": image}
```

## SaveImageNode

### Description

Save an image from the given url to disk, mainly for debugging.

### Inputs

```python
class SaveImageInput(ImageModel):
    path: str = Field("debug.png", description="The path to save the image.")
```

### Functional Outputs

```python
class EmptyOutput(BaseModel):
    pass
```

### API Outputs

*Same as the functional outputs.*

### Source Codes

```python
@Node.register("debug.save_image")
class SaveImageNode(IWithImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            SaveImageInput,
            output_model=EmptyOutput,
            description="Save an image from the given url to disk, mainly for debugging.",
        )

    async def execute(self) -> dict:
        image = await self.get_image_from("url")
        image.save(self.data["path"])
        return {}
```

## SaveImagesNode

### Description

Save images from the given urls to disk, mainly for debugging.

### Inputs

```python
class SaveImagesInput(BaseModel):
    urls: List[TImage] = Field(..., description="The urls of the images.")
    prefix: str = Field("debug", description="The prefix to save the images.")
```

### Functional Outputs

```python
class EmptyOutput(BaseModel):
    pass
```

### API Outputs

*Same as the functional outputs.*

### Source Codes

```python
@Node.register("debug.save_images")
class SaveImagesNode(IWithImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            SaveImagesInput,
            output_model=EmptyOutput,
            description="Save images from the given urls to disk, mainly for debugging.",
        )

    async def execute(self) -> dict:
        tasks = [self.fetch_image(str(v), v) for v in self.data["urls"]]
        images = await asyncio.gather(*tasks)
        prefix = self.data["prefix"]
        for i, image in enumerate(images):
            image.save(f"{prefix}_{i}.png")
        return {}
```

## BinaryOpNode

### Description

Perform binary (numpy) operations on two images.

### Inputs

```python
class BinaryOpInput(ImageModel):
    url2: TImage = Field(..., description="The second image to process.")
    op: str = Field(..., description="The (numpy) operation.")
```

### Functional Outputs

'BinaryOpNode' has following outputs:
- image

### API Outputs

```python
class ImageAPIOuput(DocModel):
    image: str = Field(..., description="The base64 encoded image.")
```

### Source Codes

```python
@Node.register("cv.binary_op")
class BinaryOpNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = BinaryOpInput
        schema.description = "Perform binary (numpy) operations on two images."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        op_name = self.data["op"]
        op = getattr(np, op_name, None)
        if op is None:
            raise ValueError(f"`{op_name}` is not a valid numpy operation")
        array = np.array(await self.get_image_from("url"))
        array2 = np.array(await self.get_image_from("url2"))
        image = Image.fromarray(op(array, array2))
        return {"image": image}
```

## BlurNode

### Description

Blur an image with a Gaussian filter.

### Inputs

```python
class BlurInput(ImageModel):
    radius: int = Field(2, description="Size of the blurring kernel.")
```

### Functional Outputs

'BlurNode' has following outputs:
- image

### API Outputs

```python
class ImageAPIOuput(DocModel):
    image: str = Field(..., description="The base64 encoded image.")
```

### Source Codes

```python
@Node.register("cv.blur")
class BlurNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = BlurInput
        schema.description = "Blur an image with a Gaussian filter."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        image = await self.get_image_from("url")
        image = image.filter(GaussianBlur(radius=self.data["radius"]))
        return {"image": image}
```

## InverseNode

### Description

Inverse an image.

### Inputs

```python
class ImageModel(DocModel):
    url: TImage = Field(..., description="The url / PIL.Image instance of the image.")
```

### Functional Outputs

'InverseNode' has following outputs:
- image

### API Outputs

```python
class ImageAPIOuput(DocModel):
    image: str = Field(..., description="The base64 encoded image.")
```

### Source Codes

```python
@Node.register("cv.inverse")
class InverseNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.description = "Inverse an image."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        image = await self.get_image_from("url")
        image = Image.fromarray(255 - np.array(image))
        return {"image": image}
```

## GrayscaleNode

### Description

Convert an image to grayscale.

### Inputs

```python
class ImageModel(DocModel):
    url: TImage = Field(..., description="The url / PIL.Image instance of the image.")
```

### Functional Outputs

'GrayscaleNode' has following outputs:
- image

### API Outputs

```python
class ImageAPIOuput(DocModel):
    image: str = Field(..., description="The base64 encoded image.")
```

### Source Codes

```python
@Node.register("cv.grayscale")
class GrayscaleNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.description = "Convert an image to grayscale."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        image = await self.get_image_from("url")
        image = image.convert("L")
        return {"image": image}
```

## ErodeAlphaNode

### Description

Erode an image.

### Inputs

```python
class ErodeAlphaInput(ImageModel):
    n_iter: int = Field(1, description="Number of iterations.")
    kernel_size: int = Field(3, description="Size of the kernel.")
    threshold: int = Field(0, description="Threshold of the alpha channel.")
    padding: int = Field(8, description="Padding for the image.")
```

### Functional Outputs

'ErodeAlphaNode' has following outputs:
- image

### API Outputs

```python
class ImageAPIOuput(DocModel):
    image: str = Field(..., description="The base64 encoded image.")
```

### Source Codes

```python
@Node.register("cv.erode_alpha")
class ErodeAlphaNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = ErodeAlphaInput
        schema.description = "Erode an image."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        n_iter = self.data["n_iter"]
        kernel_size = self.data["kernel_size"]
        threshold = self.data["threshold"]
        padding = self.data["padding"]
        image = await self.get_image_from("url")
        w, h = image.size
        array = np.array(image.convert("RGBA"))
        alpha = array[..., -1]
        padded = np.pad(alpha, (padding, padding), constant_values=0)
        binarized = cv2.threshold(padded, threshold, 255, cv2.THRESH_BINARY)[1]
        eroded = erode(binarized, n_iter, kernel_size)
        shrinked = eroded[padding:-padding, padding:-padding]
        merged_alpha = np.minimum(alpha, shrinked)
        array[..., -1] = merged_alpha
        rgb = array[..., :3].reshape([-1, 3])
        rgb[(merged_alpha == 0).ravel()] = 0
        array[..., :3] = rgb.reshape([h, w, 3])
        image = Image.fromarray(array)
        return {"image": image}
```

## ResizeNode

### Description

Resize an image.

### Inputs

```python
class ResizeInput(ResamplingModel, WHModel, ImageModel):
    mode: ResizeMode = Field(ResizeMode.FILL, description="The Resize mode.")
```

### Functional Outputs

'ResizeNode' has following outputs:
- image

### API Outputs

```python
class ImageAPIOuput(DocModel):
    image: str = Field(..., description="The base64 encoded image.")
```

### Source Codes

```python
@Node.register("cv.resize")
class ResizeNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = ResizeInput
        schema.description = "Resize an image."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        w = self.data["w"]
        h = self.data["h"]
        mode = self.data["mode"]
        resampling = self.data["resampling"]
        image = await self.get_image_from("url")
        img_w, img_h = image.size
        if mode != ResizeMode.FILL:
            w_ratio = w / img_w
            h_ratio = h / img_h
            if mode == ResizeMode.FIT:
                ratio = min(w_ratio, h_ratio)
            else:
                ratio = max(w_ratio, h_ratio)
            w = round(img_w * ratio)
            h = round(img_h * ratio)
        image = resize(image, w, h, resampling)
        return {"image": image}
```

## AffineNode

### Description

Affine an image.

### Inputs

```python
class AffineInput(ResamplingModel, WHModel, BaseAffineModel, ImageModel):
    force_rgba: bool = Field(
        False,
        description="Whether to force the output to be RGBA.",
    )
```

### Functional Outputs

'AffineNode' has following outputs:
- image

### API Outputs

```python
class ImageAPIOuput(DocModel):
    image: str = Field(..., description="The base64 encoded image.")
```

### Source Codes

```python
@Node.register("cv.affine")
class AffineNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = AffineInput
        schema.description = "Affine an image."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        image = await self.get_image_from("url")
        if self.data["force_rgba"] and image.mode != "RGBA":
            image = image.convert("RGBA")
        affine_kw = shallow_copy_dict(self.data)
        affine_kw["image"] = image
        array = safe_execute(affine, affine_kw)
        image = Image.fromarray(array)
        return {"image": image}
```

## GetMaskNode

### Description

Get the mask of an image.

### Inputs

```python
class GetMaskInput(ImageModel):
    get_inverse: bool = Field(False, description="Whether get the inverse mask.")
    binarize_threshold: Optional[int] = Field(
        None,
        ge=0,
        le=255,
        description="If not 'None', will binarize the mask with this value.",
    )
```

### Functional Outputs

'GetMaskNode' has following outputs:
- image

### API Outputs

```python
class ImageAPIOuput(DocModel):
    image: str = Field(..., description="The base64 encoded image.")
```

### Source Codes

```python
@Node.register("cv.get_mask")
class GetMaskNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = GetMaskInput
        schema.description = "Get the mask of an image."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        image = await self.get_image_from("url")
        if image.mode == "RGBA":
            mask = np.array(image)[..., -1]
        else:
            mask = np.array(image.convert("L"))
        if self.data["get_inverse"]:
            mask = 255 - mask
        binarize_threshold = self.data["binarize_threshold"]
        if binarize_threshold is not None:
            mask = np.where(mask > binarize_threshold, 255, 0)
            mask = mask.astype(np.uint8)
        mask_image = Image.fromarray(mask)
        return {"image": mask_image}
```

## FillBGNode

### Description

Fill the background of an image.

### Inputs

```python
class FillBGInput(ImageModel):
    bg: Optional[Union[int, Tuple[int, int, int]]] = Field(
        None,
        description="""Target background color.
> If not specified, 'get_contrast_bg' will be used to calculate the 'bg'.
""",
    )
```

### Functional Outputs

'FillBGNode' has following outputs:
- image

### API Outputs

```python
class ImageAPIOuput(DocModel):
    image: str = Field(..., description="The base64 encoded image.")
```

### Source Codes

```python
@Node.register("cv.fill_bg")
class FillBGNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = FillBGInput
        schema.description = "Fill the background of an image."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        image = await self.get_image_from("url")
        bg = self.data["bg"]
        if bg is None:
            bg = get_contrast_bg(image)
        if isinstance(bg, int):
            bg = bg, bg, bg
        image = Image.fromarray(np.array(image))
        image = to_rgb(image, bg)
        return {"image": image}
```

## GetSizeNode

### Description

Get the size of an image.

### Inputs

```python
class ImageModel(DocModel):
    url: TImage = Field(..., description="The url / PIL.Image instance of the image.")
```

### Functional Outputs

```python
class GetSizeOutput(BaseModel):
    size: Tuple[int, int] = Field(..., description="The size of the image.")
```

### API Outputs

*Same as the functional outputs.*

### Source Codes

```python
@Node.register("cv.get_size")
class GetSizeNode(IWithImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            ImageModel,
            GetSizeOutput,
            description="Get the size of an image.",
        )

    async def execute(self) -> Dict[str, Tuple[int, int]]:
        image = await self.get_image_from("url")
        return {"size": image.size}
```

## ModifyBoxNode

### Description

Modify the box.

### Inputs

```python
class ModifyBoxInput(LTRBModel):
    w: Optional[int] = Field(None, description="The width of the image.")
    h: Optional[int] = Field(None, description="The height of the image.")
    padding: int = Field(0, description="The padding size.")
    to_square: bool = Field(False, description="Turn the box into a square box.")
```

### Functional Outputs

```python
class LTRBModel(DocModel):
    lt_rb: Tuple[int, int, int, int] = Field(
        ...,
        description="The left-top and right-bottom points.",
    )
```

### API Outputs

*Same as the functional outputs.*

### Source Codes

```python
@Node.register("cv.modify_box")
class ModifyBoxNode(Node):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(ModifyBoxInput, LTRBModel, description="Modify the box.")

    async def execute(self) -> Dict[str, Tuple[int, int, int, int]]:
        box = ImageBox(*self.data["lt_rb"])
        box = box.pad(self.data["padding"], w=self.data["w"], h=self.data["h"])
        if self.data["to_square"]:
            box = box.to_square()
        return {"lt_rb": box.tuple}
```

## GenerateMasksNode

### Description

Generate mask images from boxes.

### Inputs

```python
class GenerateMasksInput(BaseModel):
    w: int = Field(..., description="The width of the canvas.")
    h: int = Field(..., description="The height of the canvas.")
    lt_rb_list: List[Tuple[int, int, int, int]] = Field(..., description="The boxes.")
    merge: bool = Field(False, description="Whether merge the masks.")
```

### Functional Outputs

'GenerateMasksNode' has following outputs:
- masks

### API Outputs

```python
class GenerateMaskAPIOutput(BaseModel):
    masks: List[str] = Field(..., description="The base64 encoded masks.")
```

### Source Codes

```python
@Node.register("cv.generate_masks")
class GenerateMasksNode(Node):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            GenerateMasksInput,
            api_output_model=GenerateMaskAPIOutput,
            output_names=["masks"],
            description="Generate mask images from boxes.",
        )

    @classmethod
    async def get_api_response(cls, results: Dict[str, Any]) -> Dict[str, List[str]]:
        masks = list(map(to_base64, results["masks"]))
        return {"masks": masks}

    async def execute(self) -> Dict[str, List[Image.Image]]:
        merge = self.data["merge"]
        canvas = np.zeros((self.data["h"], self.data["w"]), dtype=np.uint8)
        results = []
        for l, t, r, b in self.data["lt_rb_list"]:
            i_canvas = canvas if merge else canvas.copy()
            i_canvas[t:b, l:r] = 255
            if not merge:
                results.append(Image.fromarray(i_canvas))
        if merge:
            results.append(Image.fromarray(canvas))
        return {"masks": results}
```

## CropImageNode

### Description

Crop an image.

### Inputs

```python
class CropImageInput(LTRBModel, ImageModel):
    pass
```

### Functional Outputs

'CropImageNode' has following outputs:
- image

### API Outputs

```python
class ImageAPIOuput(DocModel):
    image: str = Field(..., description="The base64 encoded image.")
```

### Source Codes

```python
@Node.register("cv.crop_image")
class CropImageNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = CropImageInput
        schema.description = "Crop an image."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        image = await self.get_image_from("url")
        image = image.crop(self.data["lt_rb"])
        return {"image": image}
```

## PasteNode

### Description

Paste an image onto another image.

### Inputs

```python
class PasteInput(ResamplingModel, BaseAffineModel, PasteModel, ImageModel):
    pass
```

### Functional Outputs

'PasteNode' has following outputs:
- rgb
- mask
- pasted

### API Outputs

```python
class PasteAPIOutput(BaseModel):
    rgb: str = Field(
        ...,
        description="The base64 encoded RGB image.\n"
        "> This is the affined 'fg' which does not contain the background.",
    )
    mask: str = Field(
        ...,
        description="The base64 encoded mask image.\n"
        "> This is the affined 'fg''s alpha channel.",
    )
    pasted: str = Field(..., description="The base64 encoded pasted image.")
```

### Source Codes

```python
@Node.register("cv.paste")
class PasteNode(IWithImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            PasteInput,
            api_output_model=PasteAPIOutput,
            output_names=["rgb", "mask", "pasted"],
            description="Paste an image onto another image.",
        )

    @classmethod
    async def get_api_response(cls, results: Dict[str, Any]) -> Dict[str, str]:
        return {k: to_base64(v) for k, v in results.items()}

    async def execute(self) -> Dict[str, Image.Image]:
        paste_kw = shallow_copy_dict(self.data)
        paste_kw["fg"] = await self.get_image_from("url")
        paste_kw["bg"] = await self.get_image_from("bg_url")
        rgb, mask, pasted = safe_execute(paste, paste_kw)
        rgb = Image.fromarray(rgb)
        mask = Image.fromarray(mask)
        pasted = Image.fromarray(pasted)
        return {"rgb": rgb, "mask": mask, "pasted": pasted}
```

## OpenAIImageToTextNode

### Description

Image captioning with OpenAI API.

### Inputs

```python
class OpenAIImageToTextInput(ImageModel):
    prompt: Optional[str] = Field(None, description="Prompt for image captioning.")
```

### Functional Outputs

```python
class TextModel(DocModel):
    text: str = Field(..., description="The text.")
```

### API Outputs

*Same as the functional outputs.*

### Source Codes

```python
@Node.register("openai.img2txt")
class OpenAIImageToTextNode(IWithOpenAINode):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            OpenAIImageToTextInput,
            TextModel,
            description="Image captioning with OpenAI API.",
        )

    async def execute(self) -> Dict[str, str]:
        image = await self.get_image_from("url")
        image_url = await self.openai_client.upload(image)
        prompt = self.data["prompt"]
        if prompt is None:
            prompt = "Describe this image to a person with impaired vision. Be short and concise."
        caption = await self.openai_client.client.chat.completions.create(
            model="gpt-4-vision-preview",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        )
        caption = caption.choices[0].message.content
        if not caption:
            raise RuntimeError("Empty or missing caption.")
        return {"text": caption}
```

## OpenAITextToImageNode

### Description

Image generation with OpenAI API.

### Inputs

```python
class OpenAITextToImageInput(TextModel):
    size: str = Field("1024x1024", description="Image size.")
    model: str = Field("dall-e-3", description="Model name.")
    quality: str = Field("standard", description="Image quality.")
```

### Functional Outputs

```python
class OpenAITextToImageOutput(BaseModel):
    image_url: str = Field(..., description="The url of the generated image.")
```

### API Outputs

*Same as the functional outputs.*

### Source Codes

```python
@Node.register("openai.txt2img")
class OpenAITextToImageNode(IWithOpenAINode):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            OpenAITextToImageInput,
            OpenAITextToImageOutput,
            description="Image generation with OpenAI API.",
        )

    async def execute(self) -> Dict[str, str]:
        response = await self.openai_client.client.images.generate(
            model=self.data["model"],
            size=self.data["size"],
            quality=self.data["quality"],
            n=1,
            prompt=self.data["text"],
        )
        image_url = response.data[0].url
        if not image_url:
            raise RuntimeError("Empty or missing image url.")
        return {"image_url": image_url}
```


# Examples

## Coding Examples

### `examples/hello_world.py`

```python
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
```

### `examples/openai_image_variation.py`

```python
"""
Image variation example.

This example shows how to use `OpenAI` nodes to generate a variation of the given image. Please:
> - set `OPENAI_API_KEY` environment variable
> - install with `pip install carefree-workflow[openai]` or `pip install carefree-workflow[full]`

to run this example.
"""

import asyncio

from cflow import *


async def main() -> None:
    cat_url = "https://cdn.pixabay.com/photo/2016/01/20/13/05/cat-1151519_1280.jpg"
    injections = [Injection("img2txt", "text", "text")]
    flow = (
        Flow()
        .push(OpenAIImageToTextNode("img2txt", dict(url=cat_url)))
        .push(OpenAITextToImageNode("txt2img", injections=injections))
    )
    results = await flow.execute("txt2img", verbose=True)
    render_workflow(flow).save("workflow.png")
    print("> generated image url:", results["txt2img"]["image_url"])


if __name__ == "__main__":
    asyncio.run(main())
```

### `examples/download_images.py`

```python
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
```

### `examples/sleep.py`

```python
"""
Sleep example.

This example implements `SleepNode` / `AsyncSleepNode` which takes a `time` as input and returns a `message` as output.
Then, the `EchoNode` is used to print out the results from `SleepNode`.
This example also uses `render_workflow` to render the workflow graph, and `dump` to dump the workflow to a JSON file.
"""

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
```

## Workflow JSON Examples

### `examples/workflows/download_images.json`

Save images from urls.
- Adjust the `urls` and `save_prefix` in the first `node` to save different images with different prefixes.
- The images will be saved concurrently.

```json
{
  "target": "save",
  "intermediate": [],
  "verbose": true,
  "nodes": [
    {
      "key": "params",
      "type": "common.parameters",
      "data": {
        "params": {
          "urls": [
            "https://cdn.pixabay.com/photo/2016/01/20/13/05/cat-1151519_1280.jpg",
            "https://cdn.pixabay.com/photo/2020/03/31/19/20/dog-4988985_1280.jpg"
          ],
          "save_prefix": "image"
        }
      }
    },
    {
      "key": "save",
      "type": "debug.save_images",
      "injections": [
        {
          "src_key": "params",
          "src_hierarchy": "params.urls",
          "dst_hierarchy": "urls"
        },
        {
          "src_key": "params",
          "src_hierarchy": "params.save_prefix",
          "dst_hierarchy": "prefix"
        }
      ]
    }
  ]
}
```

### `examples/workflows/cv/get_blur_images.json`

Get blurred images from the given urls.
- Adjust the `urls` and `save_prefix` in the first `node` to process different images and save with different prefixes.
- Adjust the `blur_radius` in the first `node` to control the blur strength.
- The images will be processed concurrently.

```json
{
  "target": "save",
  "intermediate": [],
  "verbose": true,
  "nodes": [
    {
      "key": "params",
      "type": "common.parameters",
      "data": {
        "params": {
          "urls": [
            "https://cdn.pixabay.com/photo/2016/01/20/13/05/cat-1151519_1280.jpg",
            "https://cdn.pixabay.com/photo/2020/03/31/19/20/dog-4988985_1280.jpg"
          ],
          "blur_radius": 3,
          "save_prefix": "blur"
        }
      }
    },
    {
      "key": "download",
      "type": "common.loop",
      "injections": [
        {
          "src_key": "params",
          "src_hierarchy": "params.urls",
          "dst_hierarchy": "loop_values.url"
        }
      ],
      "data": {
        "base_node": "common.download_image",
        "extract_hierarchy": "image",
        "verbose": true
      }
    },
    {
      "key": "blur",
      "type": "common.loop",
      "injections": [
        {
          "src_key": "download",
          "src_hierarchy": "results",
          "dst_hierarchy": "loop_values.url"
        },
        {
          "src_key": "params",
          "src_hierarchy": "params.blur_radius",
          "dst_hierarchy": "base_data.radius"
        }
      ],
      "data": {
        "base_node": "cv.blur",
        "extract_hierarchy": "image",
        "verbose": true
      }
    },
    {
      "key": "save",
      "type": "debug.save_images",
      "injections": [
        {
          "src_key": "blur",
          "src_hierarchy": "results",
          "dst_hierarchy": "urls"
        },
        {
          "src_key": "params",
          "src_hierarchy": "params.save_prefix",
          "dst_hierarchy": "prefix"
        }
      ]
    }
  ]
}
```

### `examples/workflows/cv/get_blur_images_simple.json`

Get blurred image from the given url.
- Adjust the `url` and `save_prefix` in the first `node` to process different image and save with different prefix.
- Adjust the `blur_radius` in the first `node` to control the blur strength.

```json
{
  "target": "save",
  "intermediate": [],
  "verbose": true,
  "nodes": [
    {
      "key": "params",
      "type": "common.parameters",
      "data": {
        "params": {
          "url": "https://cdn.pixabay.com/photo/2016/01/20/13/05/cat-1151519_1280.jpg",
          "blur_radius": 3,
          "save_prefix": "blur"
        }
      }
    },
    {
      "key": "blur",
      "type": "cv.blur",
      "injections": [
        {
          "src_key": "params",
          "src_hierarchy": "params.url",
          "dst_hierarchy": "url"
        },
        {
          "src_key": "params",
          "src_hierarchy": "params.blur_radius",
          "dst_hierarchy": "base_data.radius"
        }
      ]
    },
    {
      "key": "save",
      "type": "debug.save_images",
      "injections": [
        {
          "src_key": "blur",
          "src_hierarchy": "image",
          "dst_hierarchy": "urls.0"
        },
        {
          "src_key": "params",
          "src_hierarchy": "params.save_prefix",
          "dst_hierarchy": "prefix"
        }
      ],
      "data": {
        "urls": []
      }
    }
  ]
}
```

### `examples/workflows/cv/get_resized_images.json`

Get resized images from the given urls.
- Adjust the `urls` and `save_prefix` in the first `node` to process different images and save with different prefixes.
- Adjust the `target_w`, `target_h` and `resize_mode` in the first `node` to control the resize behaviour.
- The images will be processed concurrently.

```json
{
  "target": "save",
  "intermediate": [],
  "verbose": true,
  "nodes": [
    {
      "key": "params",
      "type": "common.parameters",
      "data": {
        "params": {
          "urls": [
            "https://cdn.pixabay.com/photo/2016/01/20/13/05/cat-1151519_1280.jpg",
            "https://cdn.pixabay.com/photo/2020/03/31/19/20/dog-4988985_1280.jpg"
          ],
          "target_w": 512,
          "target_h": 512,
          "resize_mode": "fit",
          "save_prefix": "resized"
        }
      }
    },
    {
      "key": "download",
      "type": "common.loop",
      "injections": [
        {
          "src_key": "params",
          "src_hierarchy": "params.urls",
          "dst_hierarchy": "loop_values.url"
        }
      ],
      "data": {
        "base_node": "common.download_image",
        "extract_hierarchy": "image",
        "verbose": true
      }
    },
    {
      "key": "resize",
      "type": "common.loop",
      "injections": [
        {
          "src_key": "download",
          "src_hierarchy": "results",
          "dst_hierarchy": "loop_values.url"
        },
        {
          "src_key": "params",
          "src_hierarchy": "params.target_w",
          "dst_hierarchy": "base_data.w"
        },
        {
          "src_key": "params",
          "src_hierarchy": "params.target_h",
          "dst_hierarchy": "base_data.h"
        },
        {
          "src_key": "params",
          "src_hierarchy": "params.resize_mode",
          "dst_hierarchy": "base_data.mode"
        }
      ],
      "data": {
        "base_node": "cv.resize",
        "extract_hierarchy": "image",
        "verbose": true
      }
    },
    {
      "key": "save",
      "type": "debug.save_images",
      "injections": [
        {
          "src_key": "resize",
          "src_hierarchy": "results",
          "dst_hierarchy": "urls"
        },
        {
          "src_key": "params",
          "src_hierarchy": "params.save_prefix",
          "dst_hierarchy": "prefix"
        }
      ]
    }
  ]
}
```

### `examples/workflows/cv/get_grayscale_images.json`

Get grayscale images from the given urls.
- Adjust the `urls` and `save_prefix` in the first `node` to process different images and save with different prefixes.
- The images will be processed concurrently.

```json
{
  "target": "save",
  "intermediate": [],
  "verbose": true,
  "nodes": [
    {
      "key": "params",
      "type": "common.parameters",
      "data": {
        "params": {
          "urls": [
            "https://cdn.pixabay.com/photo/2016/01/20/13/05/cat-1151519_1280.jpg",
            "https://cdn.pixabay.com/photo/2020/03/31/19/20/dog-4988985_1280.jpg"
          ],
          "save_prefix": "grayscale"
        }
      }
    },
    {
      "key": "download",
      "type": "common.loop",
      "injections": [
        {
          "src_key": "params",
          "src_hierarchy": "params.urls",
          "dst_hierarchy": "loop_values.url"
        }
      ],
      "data": {
        "base_node": "common.download_image",
        "extract_hierarchy": "image",
        "verbose": true
      }
    },
    {
      "key": "grayscale",
      "type": "common.loop",
      "injections": [
        {
          "src_key": "download",
          "src_hierarchy": "results",
          "dst_hierarchy": "loop_values.url"
        }
      ],
      "data": {
        "base_node": "cv.grayscale",
        "extract_hierarchy": "image",
        "verbose": true
      }
    },
    {
      "key": "save",
      "type": "debug.save_images",
      "injections": [
        {
          "src_key": "grayscale",
          "src_hierarchy": "results",
          "dst_hierarchy": "urls"
        },
        {
          "src_key": "params",
          "src_hierarchy": "params.save_prefix",
          "dst_hierarchy": "prefix"
        }
      ]
    }
  ]
}
```

### `examples/workflows/openai/text2image2text.json`

Perform text -> image -> text process from the given prompts.
- Adjust the `prompts` in the first `node` for different processes.
- The processes will be launched concurrently.

```json
{
  "target": "img2txt",
  "intermediate": [],
  "verbose": true,
  "nodes": [
    {
      "key": "params",
      "type": "common.parameters",
      "data": {
        "params": {
          "prompts": [
            "A lovely little cat.",
            "A lovely little dog."
          ]
        }
      }
    },
    {
      "key": "txt2img",
      "type": "common.loop",
      "injections": [
        {
          "src_key": "params",
          "src_hierarchy": "params.prompts",
          "dst_hierarchy": "loop_values.text"
        }
      ],
      "data": {
        "base_node": "openai.txt2img",
        "extract_hierarchy": "image_url",
        "verbose": true
      }
    },
    {
      "key": "img2txt",
      "type": "common.loop",
      "injections": [
        {
          "src_key": "txt2img",
          "src_hierarchy": "results",
          "dst_hierarchy": "loop_values.url"
        }
      ],
      "data": {
        "base_node": "openai.img2txt",
        "extract_hierarchy": "text",
        "verbose": true
      }
    }
  ]
}
```

### `examples/workflows/openai/get_image_generations.json`

Get image generations from the given prompts.
- Adjust the `prompts` and `save_prefix` in the first `node` to generate different images and save with different prefixes.
- The images will be generated concurrently.

```json
{
  "target": "save",
  "intermediate": [],
  "verbose": true,
  "nodes": [
    {
      "key": "params",
      "type": "common.parameters",
      "data": {
        "params": {
          "prompts": [
            "A lovely little cat.",
            "A lovely little dog."
          ],
          "save_prefix": "openai_txt2img"
        }
      }
    },
    {
      "key": "txt2img",
      "type": "common.loop",
      "injections": [
        {
          "src_key": "params",
          "src_hierarchy": "params.prompts",
          "dst_hierarchy": "loop_values.text"
        }
      ],
      "data": {
        "base_node": "openai.txt2img",
        "extract_hierarchy": "image_url",
        "verbose": true
      }
    },
    {
      "key": "save",
      "type": "debug.save_images",
      "injections": [
        {
          "src_key": "txt2img",
          "src_hierarchy": "results",
          "dst_hierarchy": "urls"
        },
        {
          "src_key": "params",
          "src_hierarchy": "params.save_prefix",
          "dst_hierarchy": "prefix"
        }
      ]
    }
  ]
}
```

### `examples/workflows/openai/image2text2image_simple.json`

Perform image -> text -> image process from the given image.
- Adjust the `url` and `save_prefix` in the first `node` for a different process and save with different prefixes.

```json
{
  "target": "save",
  "intermediate": [],
  "verbose": true,
  "nodes": [
    {
      "key": "params",
      "type": "common.parameters",
      "data": {
        "params": {
          "url": "https://cdn.pixabay.com/photo/2016/01/20/13/05/cat-1151519_1280.jpg",
          "save_prefix": "openai_img2txt2img"
        }
      }
    },
    {
      "key": "img2txt",
      "type": "openai.img2txt",
      "injections": [
        {
          "src_key": "params",
          "src_hierarchy": "params.url",
          "dst_hierarchy": "url"
        }
      ]
    },
    {
      "key": "txt2img",
      "type": "openai.txt2img",
      "injections": [
        {
          "src_key": "img2txt",
          "src_hierarchy": "text",
          "dst_hierarchy": "text"
        }
      ]
    },
    {
      "key": "save",
      "type": "debug.save_images",
      "injections": [
        {
          "src_key": "txt2img",
          "src_hierarchy": "image_url",
          "dst_hierarchy": "urls.0"
        },
        {
          "src_key": "params",
          "src_hierarchy": "params.save_prefix",
          "dst_hierarchy": "prefix"
        }
      ],
      "data": {
        "urls": []
      }
    }
  ]
}
```

### `examples/workflows/openai/text2image2text_simple.json`

Perform text -> image -> text process from the given prompt.
- Adjust the `prompt` in the first `node` for a different process.

```json
{
  "target": "img2txt",
  "intermediate": [],
  "verbose": true,
  "nodes": [
    {
      "key": "params",
      "type": "common.parameters",
      "data": {
        "params": {
          "prompt": "A lovely little cat."
        }
      }
    },
    {
      "key": "txt2img",
      "type": "openai.txt2img",
      "injections": [
        {
          "src_key": "params",
          "src_hierarchy": "params.prompt",
          "dst_hierarchy": "text"
        }
      ]
    },
    {
      "key": "img2txt",
      "type": "openai.img2txt",
      "injections": [
        {
          "src_key": "txt2img",
          "src_hierarchy": "image_url",
          "dst_hierarchy": "url"
        }
      ]
    }
  ]
}
```

### `examples/workflows/openai/get_image_captionings.json`

Get image captionings from the given urls.
- Adjust the `urls` in the first `node` to process different images.
- The images will be processed concurrently.

```json
{
  "target": "img2txt",
  "intermediate": [],
  "verbose": true,
  "nodes": [
    {
      "key": "params",
      "type": "common.parameters",
      "data": {
        "params": {
          "urls": [
            "https://cdn.pixabay.com/photo/2016/01/20/13/05/cat-1151519_1280.jpg",
            "https://cdn.pixabay.com/photo/2020/03/31/19/20/dog-4988985_1280.jpg"
          ]
        }
      }
    },
    {
      "key": "download",
      "type": "common.loop",
      "injections": [
        {
          "src_key": "params",
          "src_hierarchy": "params.urls",
          "dst_hierarchy": "loop_values.url"
        }
      ],
      "data": {
        "base_node": "common.download_image",
        "extract_hierarchy": "image",
        "verbose": true
      }
    },
    {
      "key": "img2txt",
      "type": "common.loop",
      "injections": [
        {
          "src_key": "download",
          "src_hierarchy": "results",
          "dst_hierarchy": "loop_values.url"
        }
      ],
      "data": {
        "base_node": "openai.img2txt",
        "extract_hierarchy": "text",
        "verbose": true
      }
    }
  ]
}
```
