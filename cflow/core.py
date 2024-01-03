import json
import time
import asyncio

from abc import abstractmethod
from abc import ABCMeta
from cftool import console
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Optional
from pathlib import Path
from pydantic import BaseModel
from dataclasses import field
from dataclasses import dataclass
from cftool.web import get_err_msg
from cftool.misc import offload
from cftool.misc import random_hash
from cftool.misc import register_core
from cftool.misc import shallow_copy_dict
from cftool.misc import ISerializableDataClass
from cftool.data_structures import Item
from cftool.data_structures import Bundle


TPath = Union[str, Path]
TTNode = TypeVar("TTNode", bound=Type["Node"])
nodes: Dict[str, Type["Node"]] = {}

UNDEFINED_PLACEHOLDER = "$undefined$"
EXCEPTION_MESSAGE_KEY = "$exception$"
ALL_LATENCIES_KEY = "$all_latencies$"

GATHER_NODE = "common.gather"


def extract_from(data: Any, hierarchy: str) -> Any:
    hierarchies = hierarchy.split(".")
    for h in hierarchies:
        if isinstance(data, list):
            try:
                ih = int(h)
            except:
                msg = f"current value is list, but '{h}' is not int"
                raise ValueError(msg)
            data = data[ih]
        elif isinstance(data, dict):
            data = data[h]
        else:
            raise ValueError(
                f"hierarchy '{h}' is required but current value type "
                f"is '{type(data)}' ({data})"
            )
    return data


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
        - `src_hierarchy` can be very complex:
          - use `int` as `list` index, and `str` as `dict` key.
          - use `.` to represent nested structure.
          - for example, you can use `a.0.b` to indicate `results["a"][0]["b"]`.
        - If `None`, all results of the dependent node will be used.
    dst_hierarchy : str
        The 'dst_hierarchy' of the current node's `data`.
        - `dst_hierarchy` can be very complex:
          - use `int` as `list` index, and `str` as `dict` key.
          - use `.` to represent nested structure.
          - for example, if you want to inject to `data["a"][0]["b"]`, you can use `a.0.b`
          as the `dst_hierarchy`.

    """

    src_key: str
    src_hierarchy: Optional[str]
    dst_hierarchy: str


@dataclass
class Schema:
    """
    A class that represents a Schema of a node.

    Implement `get_schema` method and return a `Schema` instance for your nodes
    can help us auto-generate UIs, APIs and documents.

    Attributes
    ----------
    input_model : Optional[Type[BaseModel]]
        The input data model of the node.
        > If your inputs are not JSON serializable, you can use `input_names` instead.
    output_model : Optional[Type[BaseModel]]
        The output data model of the node.
        > If your outputs are not JSON serializable, you can use either `api_output_model`
        or `output_names` instead.
    api_output_model : Optional[Type[BaseModel]]
        The API response data model of the node.
        > This is helpful when your outputs are not JSON serializable, and you implement
        the `get_api_response` method to convert the outputs to API responses.
        > In this case, `api_output_model` should be the data model of the results returned
        by `get_api_response`.
    input_names : Optional[List[str]]
        The names of the inputs of the node.
        > This is helpful if you want to make things simple.
        > Please make sure that the input `data` of the node has exactly the same keys as `input_names`.
    output_names : Optional[List[str]]
        The names of the outputs of the node.
        > This is helpful if you want to make things simple.
        > Please make sure that the output `results` of the node has exactly the same keys as `output_names`.
    description : Optional[str]
        A description of the node.
        > This will be displayed in the auto-generated UIs / documents.

    """

    input_model: Optional[Type[BaseModel]] = None
    output_model: Optional[Type[BaseModel]] = None
    api_output_model: Optional[Type[BaseModel]] = None
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None
    description: Optional[str] = None


class Hook:
    @classmethod
    async def initialize(cls, node: "Node", flow: "Flow") -> None:
        pass

    @classmethod
    async def cleanup(cls, node: "Node") -> None:
        pass


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
        Implement this method to handle complex API responses (e.g. `PIL.Image`).
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

    # optional

    @classmethod
    def get_schema(cls) -> Optional[Schema]:
        return None

    @classmethod
    async def get_api_response(cls, results: Dict[str, Any]) -> Any:
        return results

    @classmethod
    def get_hooks(cls) -> List[Type[Hook]]:
        return []

    async def initialize(self, flow: "Flow") -> None:
        for hook in self.get_hooks():
            await hook.initialize(self, flow)

    async def cleanup(self) -> None:
        for hook in self.get_hooks():
            await hook.cleanup(self)

    # abstract

    @abstractmethod
    async def execute(self) -> Any:
        pass

    # internal

    @classmethod
    def d(cls) -> Dict[str, Type["Node"]]:
        return nodes

    @classmethod
    def register(cls, name: str) -> Callable[[TTNode], TTNode]:
        def before(cls_: Type) -> None:
            if name == "workflow":
                raise RuntimeError(
                    "`workflow` is a reserved name, please use another name "
                    f"when registering node '{cls_.__name__}'"
                )
            cls_.__identifier__ = name

        return register_core(
            name,
            cls.d(),
            allow_duplicate=False,
            before_register=before,
        )

    def copy(self) -> "Node":
        copied = self.__class__()
        copied.from_info(shallow_copy_dict(self.to_info()))
        copied.shared_pool = self.shared_pool
        return copied

    def to_item(self) -> Item["Node"]:
        if self.key is None:
            raise ValueError("node key cannot be None")
        return Item(self.key, self)

    def to_info(self) -> Dict[str, Any]:
        pool, self.shared_pool = self.shared_pool, {}
        info = super().to_info()
        info.pop("executing")
        info.pop("shared_pool")
        self.shared_pool = pool
        return info

    def from_info(self, info: Dict[str, Any]) -> None:
        super().from_info(info)
        self.injections = [Injection(**d) for d in self.injections]  # type: ignore
        if self.key is None:
            raise ValueError("node key cannot be None")
        if "." in self.key:
            raise ValueError("node key cannot contain '.'")

    def check_inputs(self) -> None:
        if not isinstance(self.data, dict):
            msg = f"input `data` ({self.data}) of node '{self.key}' should be a `dict`"
            raise ValueError(msg)
        schema = self.get_schema()
        if schema is None:
            return
        if schema.input_model is not None:
            try:
                narrowed = schema.input_model(**self.data)
                self.data = narrowed.model_dump()
            except Exception as err:
                msg = f"input data ({self.data}) does not match the schema model ({schema.input_model})"
                raise ValueError(msg) from err
        elif schema.input_names is not None:
            data_inputs = set(self.data.keys())
            schema_inputs = set(schema.input_names)
            if data_inputs != schema_inputs:
                msg = f"input data ({self.data}) does not match the schema names ({schema.input_names})"
                raise ValueError(msg)

    def check_injections(self) -> None:
        history: Dict[str, Injection] = {}
        for injection in self.injections:
            existing = history.get(injection.dst_hierarchy)
            if existing is not None:
                raise ValueError(
                    f"`dst_hierarchy` of current injection ({injection}) is duplicated "
                    f"with previous injection ({existing})"
                )
            history[injection.dst_hierarchy] = injection

    def fetch_injections(self, all_results: Dict[str, Any]) -> None:
        def inject_leaf_data(data: Any, hierarchies: List[str], value: Any) -> None:
            if len(hierarchies) == 0:
                return data
            h = hierarchies.pop(0)
            is_leaf = len(hierarchies) == 0
            if isinstance(data, list):
                try:
                    ih = int(h)
                except:
                    raise ValueError(f"current value is list, but '{h}' is not int")
                if len(data) <= ih:
                    replace_msg = "target value" if is_leaf else "an empty `dict`"
                    console.warn(
                        "current data is a list but its length is not enough, corresponding "
                        f"index ({h}) will be set to {replace_msg}, and other elements "
                        "will be set to `undefined`"
                    )
                    data.extend([UNDEFINED_PLACEHOLDER] * (ih - len(data) + 1))
                if is_leaf:
                    data[ih] = value
                else:
                    data[ih] = {}
                    inject_leaf_data(data[ih], hierarchies, value)
            elif isinstance(data, dict):
                if is_leaf:
                    data[h] = value
                else:
                    if h not in data:
                        console.warn(
                            "current data is a dict but it does not have the corresponding "
                            f"key ('{h}'), it will be set to an empty `dict`"
                        )
                        data[h] = {}
                    inject_leaf_data(data[h], hierarchies, value)
            else:
                raise ValueError(
                    f"hierarchy '{h}' is required but current value type "
                    f"is '{type(data)}' ({data})"
                )

        for injection in self.injections:
            src_key = injection.src_key
            src_out = all_results.get(src_key)
            if src_out is None:
                raise ValueError(f"cannot find cache for '{src_key}'")
            if injection.src_hierarchy is not None:
                src_out = extract_from(src_out, injection.src_hierarchy)
            dst_hierarchies = injection.dst_hierarchy.split(".")
            inject_leaf_data(self.data, dst_hierarchies, src_out)

    def check_undefined(self) -> None:
        def check(data: Any) -> None:
            if isinstance(data, list):
                for item in data:
                    check(item)
            elif isinstance(data, dict):
                for v in data.values():
                    check(v)
            elif data == UNDEFINED_PLACEHOLDER:
                raise ValueError(f"undefined value found in '{self.data}'")

        check(self.data)

    def check_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(results, dict):
            msg = f"output results ({results}) of node '{self.key}' should be a `dict`"
            raise ValueError(msg)
        schema = self.get_schema()
        if schema is None:
            return results
        if schema.output_model is not None:
            try:
                narrowed = schema.output_model(**results)
                return narrowed.model_dump()
            except Exception as err:
                msg = f"output data ({results}) does not match the schema model ({schema.output_model})"
                raise ValueError(msg) from err
        if schema.output_names is not None:
            node_outputs = set(results.keys())
            schema_outputs = set(schema.output_names)
            if node_outputs != schema_outputs:
                msg = f"output data ({results}) does not match the schema names ({schema.output_names})"
                raise ValueError(msg)
        return results

    def check_api_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        schema = self.get_schema()
        if schema is None:
            return results
        if schema.api_output_model is not None:
            try:
                narrowed = schema.api_output_model(**results)
                return narrowed.model_dump()
            except Exception as err:
                msg = f"API response ({results}) does not match the schema model ({schema.api_output_model})"
                raise ValueError(msg) from err
        return results


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

    def __init__(self, *, no_mapping: bool = False) -> None:
        super().__init__(no_mapping=no_mapping)
        self.latest_latencies: Dict[str, Dict[str, float]] = {}

    def __str__(self) -> str:
        body = ",\n  ".join(str(item.data) for item in self)
        return f"""Flow([
  {body}
])"""

    __repr__ = __str__

    def push(self, node: Node) -> "Flow":
        if node.key is None:
            raise ValueError("node key cannot be None")
        super().push(node.to_item())
        return self

    def gather(self, *targets: str) -> str:
        gather_key = f"$gather_{random_hash()[:4]}"
        injections = [Injection(k, None, k) for k in targets]
        self.push(
            Node.make(
                GATHER_NODE,
                dict(key=gather_key, injections=injections),
            )
        )
        return gather_key

    def to_json(self) -> List[Dict[str, Any]]:
        return [item.data.to_pack().asdict() for item in self]

    @classmethod
    def from_json(cls, data: List[Dict[str, Any]]) -> "Flow":
        workflow = cls()
        for pack in data:
            workflow.push(Node.from_pack(pack))
        return workflow

    def copy(self) -> "Flow":
        return Flow.from_json(self.to_json())

    def dump(self, path: TPath) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load(cls, path: TPath) -> "Flow":
        with open(path, "r") as f:
            return cls.from_json(json.load(f))

    def get_reachable(self, target: str) -> Set[str]:
        def dfs(key: str, is_target: bool) -> None:
            if not is_target and key == target:
                raise ValueError(f"cyclic dependency detected when dfs from '{target}'")
            if key in reachable:
                return
            reachable.add(key)
            item = self.get(key)
            if item is None:
                raise ValueError(
                    f"cannot find node '{key}', which is declared as a dependency, "
                    f"in the workflow ({self})"
                )
            node = item.data
            for injection in node.injections:
                dfs(injection.src_key, False)

        reachable: Set[str] = set()
        dfs(target, True)
        return reachable

    async def run(
        self,
        item: Item[Node],
        api_results: Dict[str, Any],
        all_results: Dict[str, Any],
        return_api_response: bool,
        verbose: bool,
        all_latencies: Dict[str, Dict[str, float]],
    ) -> None:
        if item.key in all_results:
            return
        start_t = time.time()
        while not all(i.src_key in all_results for i in item.data.injections):
            await asyncio.sleep(0)
        if item.data.lock_key is not None:
            while not all(
                not other.data.executing or other.data.lock_key != item.data.lock_key
                for other in self
            ):
                await asyncio.sleep(0)
        item.data.executing = True
        t0 = time.time()
        node: Node = item.data.copy()
        node.fetch_injections(all_results)
        node.check_undefined()
        node.check_inputs()
        t1 = time.time()
        if verbose:
            console.debug(f"executing node '{item.key}'")
        if not node.offload:
            results = await node.execute()
        else:
            results = await offload(node.execute())
        results = node.check_results(results)
        all_results[item.key] = results
        if return_api_response:
            results = await node.get_api_response(results)
            results = node.check_api_results(results)
            api_results[item.key] = results
        t2 = time.time()
        item.data.executing = False
        all_latencies[item.key] = dict(
            pending=t0 - start_t,
            inject=t1 - t0,
            execute=t2 - t1,
            latency=t2 - t0,
        )
        if verbose:
            console.debug(f"finished executing node '{item.key}'")

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
        Executes the workflow ending at the `target` node.

        Parameters
        ----------
        target : str
            The key of the target node which the execution will end at.
        intermediate : List[str], optional
            A list of intermediate nodes that will be returned. Default is `None`.
            - Only useful when `return_api_response` is `True`.
            - If `None`, no intermediate nodes will be returned.
        return_if_exception : bool, optional
            If `True`, the function will return even if an exception occurs. Default is `False`.
        return_api_response : bool, optional
            If `True`, the function will:
            - Only return the results of the `target` node & the `intermediate` nodes.
            - Call `get_api_response` on the results to get the final API response.
        verbose : bool, optional
            If `True`, the function will print detailed logs. Default is `False`.

        Returns
        -------
        dict
            A dictionary containing the results of the execution.
            - If `return_api_response` is `True`, only outputs of the `target` node can be accessed
            (via `results[target]`).
            - Otherwise, outputs of all nodes can be accessed (via `results[key]`, where `key` is
            the key of the node).
            - If an exception occurs during the execution, the dictionary will contain
            a key 'EXCEPTION_MESSAGE_KEY' with the error message as the value.

        """

        api_results: Dict[str, Any] = {}
        all_results: Dict[str, Any] = {}
        extra_results: Dict[str, Any] = {}
        all_latencies: Dict[str, Dict[str, float]] = {}
        if intermediate is None:
            intermediate = []
        try:
            workflow = self.copy()
            reachable = workflow.get_reachable(target)
            shared_pool: Dict[str, Any] = {}
            if target not in workflow:
                raise ValueError(f"cannot find target '{target}' in the workflow")
            reachable_nodes = [item.data for item in workflow if item.key in reachable]
            for node in reachable_nodes:
                node.check_injections()
                node.shared_pool = shared_pool
                if verbose:
                    console.debug(f"initializing node '{node.key}'")
                await node.initialize(workflow)
            await asyncio.gather(
                *(
                    workflow.run(
                        item,
                        api_results,
                        all_results,
                        return_api_response
                        and (item.key == target or item.key in intermediate),
                        verbose,
                        all_latencies,
                    )
                    for item in workflow
                    if item.key in reachable
                )
            )
            for node in reachable_nodes:
                if verbose:
                    console.debug(f"cleaning up node '{node.key}'")
                await node.cleanup()
            extra_results[EXCEPTION_MESSAGE_KEY] = None
        except Exception as err:
            if not return_if_exception:
                raise
            err_msg = get_err_msg(err)
            extra_results[EXCEPTION_MESSAGE_KEY] = err_msg
            if verbose:
                console.error(err_msg)
        self.latest_latencies = all_latencies
        extra_results[ALL_LATENCIES_KEY] = all_latencies
        final_results = api_results if return_api_response else all_results
        final_results.update(extra_results)
        return final_results


__all__ = [
    "Injection",
    "Schema",
    "Node",
    "Flow",
    "BaseModel",
]
