from typing import Any
from typing import Dict
from cftool.misc import OPTBase


class OPTClass(OPTBase):
    focus: str

    @property
    def env_key(self) -> str:
        return "CFLOW_ENV"

    @property
    def defaults(self) -> Dict[str, Any]:
        return dict(
            focus="",
        )


OPT = OPTClass()


__all__ = [
    "OPT",
]
