from typing import Any
from typing import Dict
from cftool.misc import OPTBase


class OPTClass(OPTBase):
    focus: str
    verbose: bool
    # ai nodes settings
    api_pool_limit: int
    num_control_pool: int
    use_controlnet: bool
    use_controlnet_annotator: bool
    sd_weights_pool_limit: int

    @property
    def env_key(self) -> str:
        return "CFLOW_ENV"

    @property
    def defaults(self) -> Dict[str, Any]:
        return dict(
            focus="",
            verbose=True,
            api_pool_limit=3,
            num_control_pool=3,
            use_controlnet=False,
            use_controlnet_annotator=False,
            sd_weights_pool_limit=3,
        )


OPT = OPTClass()


__all__ = [
    "OPT",
]
