import json
import click
import cflow
import asyncio
import uvicorn


api = cflow.API()


async def run_execute(file: str, *, raw: bool = False, output: str = "") -> None:
    with open(file, "r") as f:
        data = cflow.WorkflowModel(**json.load(f))
    flow = data.get_workflow()
    results = await data.run(flow, return_api_response=not raw)
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


@click.group()
def main() -> None:
    pass


@main.command()
@click.option(
    "--app",
    default="cflow.cli:api.app",
    show_default=True,
    type=str,
    help="The target app of the server.",
)
@click.option(
    "-h",
    "--host",
    default="0.0.0.0",
    show_default=True,
    type=str,
    help="The target host of the server.",
)
@click.option(
    "-p",
    "--port",
    default=8123,
    show_default=True,
    type=int,
    help="The target port of the server.",
)
@click.option(
    "-r",
    "--reload",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="Whether to reload the server when the code changes.",
)
def serve(*, app: str, host: str, port: int, reload: bool) -> None:
    uvicorn.run(app, host=host, port=port, reload=reload)


@main.command()
@click.option(
    "-f",
    "--file",
    type=str,
    help="The workflow JSON file to be executed. It should follow the `WorkflowModel` schema.",
)
@click.option(
    "-o",
    "--output",
    default="",
    show_default=True,
    type=str,
    help="The output file path, please make sure that the outputs "
    "are JSON serializable if this option is specified.",
)
@click.option(
    "--raw",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="Whether to return the raw results instead of the API responses.",
)
def execute(*, file: str, output: str, raw: bool) -> None:
    asyncio.run(run_execute(file, raw=raw, output=output))


@main.command()
@click.option(
    "-f",
    "--file",
    type=str,
    help="The workflow JSON file to be executed. It should follow the `WorkflowModel` schema.",
)
@click.option(
    "-o",
    "--output",
    type=str,
    help="The output PNG file path.",
)
def render(*, file: str, output: str) -> None:
    with open(file, "r") as f:
        data = cflow.WorkflowModel(**json.load(f))
    rendered = cflow.render_workflow(data.get_workflow(), target=data.target)
    rendered.save(output)


@main.command()
@click.option(
    "-o",
    "--output",
    default="docs.md",
    show_default=True,
    type=str,
    help="The output markdown file path, should end with `.md`.",
)
@click.option(
    "--rag",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="Whether to make the generated document more 'RAG (Retrieval-Augmented Generation) friendly'.",
)
def docs(*, output: str, rag: bool) -> None:
    cflow.generate_documents(output, rag)


__all__ = [
    "run_execute",
]
