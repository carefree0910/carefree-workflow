# carefree-workflow

`carefree-workflow` is a lightweight library for building arbitray workflows with Python.


## Agent Friendly

With the rapid development of AI (namely, the GPTs), I think modern libraries should be **agent-friendly**, which means developers should be able to build a GPT-agent on top of their libraries **automatically**, and after which users should be able to learn the library by interacting with this GPT-agent via natural language. With this in mind, `carefree-workflow` is designed to be modular, extensible, automated and 'RAG friendly'. See [Highlights](#highlights) for more details.


## Highlights

- **Async**: `async` is by design.
- **Parallel**: nodes can be executed in parallel.
- **Powerful**: complex locks / logics / dependencies can be handled.
  - You can even perform a loop with loop backs in the workflow!
- **Automated**:
  - All nodes, as well as the workflow itself, can be **automatically** turned into RESTful APIs.
  - Detailed [documentation](https://github.com/carefree0910/carefree-workflow/tree/main/docs.md) of the design / nodes / workflow / ... can be **automatically** generated, which makes this library and its extended versions agent-friendly.
    - That is to say, you can build a GPT-agent on top of this library by simply feed the auto-generated documentation ([this](https://github.com/carefree0910/carefree-workflow/tree/main/docs.md)) to it. After which, you can interact with the agent via natural language and it will tell you how to build the workflow you want (it may even be able to give you the final workflow JSON directly)!
    - We even provide a 'RAG friendly' version of the documentation ([this](https://github.com/carefree0910/carefree-workflow/tree/main/docs_rag.md)), which makes **R**etrieval-**A**ugmented **G**eneration easier. This version uses `__RAG__` as the special separator, so you can chunk the documentation into suitable parts for RAG.
- **Extensible**: you can easily extend the library with your own nodes.
- **Serializable**: the workflow can be serialized into / deserialized from a single JSON file.
- **Human Readable**: the workflow JSON file is human readable and easy to understand.
- **Lightweight**: the library is lightweight (core implementation is ~500 lines of code in a single file `core.py`) and easy to use.


## Installation

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


## Usages

### CLI

`carefree-workflow` is installed with a **C**ommand **L**ine **I**nterface (CLI) tool called `cflow`, which can help you execute / render workflows easily:

> Workflow JSON examples can be found in [`examples/workflows`](https://github.com/carefree0910/carefree-workflow/tree/main/examples/workflows).

```bash
cflow execute -f <workflow_json_file>
```

or

```bash
cflow render -f <workflow_json_file>
```

Detailed information can be found by:

```bash
cflow --help
```

### Code

Since `carefree-workflow` is extensible, you can easily extend the library with your own nodes:

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

> More code snippets can be found in [`examples`](https://github.com/carefree0910/carefree-workflow/tree/main/examples).

### Documentation

`carefree-workflow` can automatically generate documentation for you:

```bash
cflow docs -o <document_output_file>
```

> We also include the auto-generated documentation in this repository, see [`docs.md`](https://github.com/carefree0910/carefree-workflow/tree/main/docs.md).

And we have designed this auto-generated documentation to be agent-friendly, which means you can build a GPT-agent on top of this library by simply feed the auto-generated documentation to it. After which, you can interact with the agent via natural language and it will tell you how to build the workflow you want (it may even be able to give you the final workflow JSON directly)!


## Serving

`carefree-workflow` is designed to be servable, here are some rules:

- Only nodes with `get_schema` implemented can be automatically served.
  - See [`cflow/nodes/cv.py`](https://github.com/carefree0910/carefree-workflow/tree/main/cflow/nodes/cv.py) for some examples.
- Complex results (e.g. `PIL.Image`) can be handled if `get_api_response` is implemented.
  - See our [template](https://github.com/carefree0910/carefree-workflow/tree/main/cflow/nodes/templates/complex.py) if you are interested!
- The registered name will be turned into the API endpoint, with the dots (`.`) being replaced by slashes (`/`).
  - e.g. if the node is registered with `Node.register("foo.bar")`, the corresponding API endpoint will be `/foo/bar`.

You can use our CLI to launch a server under default settings easily:

```bash
cflow serve
```

More options can be found by `cflow serve --help`.


## Examples

- Code snippets can be found in [`examples`](https://github.com/carefree0910/carefree-workflow/tree/main/examples).
- Custom node templates can be found in [`cflow/nodes/templates`](https://github.com/carefree0910/carefree-workflow/tree/main/cflow/nodes/templates).
  - And some pre-implemented node examples can be found in [`cflow/nodes/cv.py`](https://github.com/carefree0910/carefree-workflow/tree/main/cflow/nodes/cv.py).
- Workflow JSON files can be found in [`examples/workflows`](https://github.com/carefree0910/carefree-workflow/tree/main/examples/workflows).
  - These JSON files can be executed by `cflow execute -f <workflow_json_file>`.
  - More execution options can be found by `cflow execute --help`.


## License

`carefree-workflow` is MIT licensed, as found in the [`LICENSE`](https://github.com/carefree0910/carefree-workflow/blob/main/LICENSE) file.

---
