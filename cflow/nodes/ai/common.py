import os
import re
import json
import secrets

import numpy as np

from PIL import Image
from enum import Enum
from cftool import console
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from pydantic import Field
from pydantic import BaseModel
from cftool.cv import to_uint8
from cftool.misc import shallow_copy_dict
from cftool.types import TNumberPair

from ..common import to_endpoint
from ..schema import TImage
from ..schema import TextModel
from ..schema import ImageModel
from ..schema import IWithHttpSessionNode
from ...parameters import OPT

try:
    from cflearn.api import APIPool
    from cflearn.zoo import SDVersions
    from cflearn.toolkit import download_checkpoint
    from cflearn.api.cv import TranslatorAPI
    from cflearn.api.multimodal import DiffusionAPI
    from cflearn.api.multimodal import InpaintingMode
    from cflearn.api.multimodal import InpaintingSettings
    from cflearn.api.multimodal import ControlledDiffusionAPI
    from cflearn.modules.multimodal.diffusion import StableDiffusion
    from cflearn.api.cv.third_party.isnet import ISNetAPI
    from cflearn.api.multimodal.third_party.blip import BLIPAPI
except:

    class SDVersions(str, Enum):  # type: ignore
        v1_5 = "v1.5"

    class InpaintingMode(str, Enum):  # type: ignore
        NORMAL = "normal"

    APIPool = None
    TranslatorAPI = None
    ImageHarmonizationAPI = None
    DiffusionAPI = None
    ControlledDiffusionAPI = None
    LaMaAPI = None
    ISNetAPI = None
    PromptEnhanceAPI = None
    BLIPAPI = None


BaseSDTag = "_base_sd"
sd_txt2img_name = "ai.txt2img.sd"
sd_img2img_name = "ai.img2img.sd"
sd_controlnet_name = "ai.controlnet"

api_pool: APIPool = None


# managements


class APIs(str, Enum):
    SD = "sd"
    SD_INPAINTING = "sd_inpainting"
    INPAINTING = "inpainting"
    ESR = "esr"
    ESR_ANIME = "esr_anime"
    ISNET = "isnet"
    BLIP = "blip"


def _base_sd_path() -> str:
    return str(download_checkpoint("ldm_sd_v1.5"))


def get_api_pool() -> APIPool:
    global api_pool
    if api_pool is None:
        api_pool = APIPool(OPT.api_pool_limit)
    return api_pool


def init_sd(**kwargs: Any) -> ControlledDiffusionAPI:
    version = SDVersions.v1_5
    kw = shallow_copy_dict(kwargs)
    kw["num_pool"] = OPT.num_control_pool
    kw["force_not_lazy"] = True
    m = ControlledDiffusionAPI.from_sd(version, **kw)
    if not OPT.use_controlnet:
        m.disable_control()
    if any(
        re.search(OPT.focus, to_endpoint(name))
        for name in [sd_txt2img_name, sd_img2img_name, sd_controlnet_name]
    ):
        m.sd_weights.limit = OPT.sd_weights_pool_limit
        m.current_sd_version = version
        console.log("> registering base sd")
        m.prepare_sd([version])
        m.sd_weights.register(BaseSDTag, _base_sd_path())
        if OPT.use_controlnet:
            console.log("> warmup ControlNet")
            m.switch_control(*m.preset_control_hints)
    if OPT.use_controlnet_annotator:
        console.log("> prepare ControlNet Annotators")
        m.prepare_annotators()
    return m


def register_sd() -> None:
    if APIPool is None:
        return
    get_api_pool().register(APIs.SD, init_sd)


def register_inpainting() -> None:
    api_pool.register(APIs.INPAINTING, DiffusionAPI.from_inpainting)


def register_esr() -> None:
    get_api_pool().register(APIs.ESR, TranslatorAPI.from_esr)


def register_esr_anime() -> None:
    get_api_pool().register(APIs.ESR_ANIME, TranslatorAPI.from_esr_anime)


def register_isnet() -> None:
    get_api_pool().register(APIs.ISNET, ISNetAPI)


def register_blip() -> None:
    get_api_pool().register(APIs.BLIP, BLIPAPI)


# enums


class SDSamplers(str, Enum):
    DDIM = "ddim"
    PLMS = "plms"
    KLMS = "klms"
    SOLVER = "solver"
    K_EULER = "k_euler"
    K_EULER_A = "k_euler_a"
    K_HEUN = "k_heun"
    K_DPMPP_2M = "k_dpmpp_2m"
    LCM = "lcm"


class SDInpaintingVersions(str, Enum):
    v1_5 = "v1.5"


class SigmasScheduler(str, Enum):
    KARRAS = "karras"


# data models


class CallbackModel(BaseModel):
    callback_url: str = Field("", description="callback url to post to")


class UseAuditModel(BaseModel):
    use_audit: bool = Field(False, description="Whether audit the outputs.")


class MaxWHModel(BaseModel):
    max_wh: int = Field(1024, description="The maximum resolution.")


class VariationModel(BaseModel):
    seed: int = Field(..., description="Seed of the variation.")
    strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Strength of the variation.",
    )


class TomeInfoModel(BaseModel):
    enable: bool = Field(False, description="Whether enable tomesd.")
    ratio: float = Field(0.5, description="The ratio of tokens to merge.")
    max_downsample: int = Field(
        1,
        description="Apply ToMe to layers with at most this amount of downsampling.",
    )
    sx: int = Field(2, description="The stride for computing dst sets.")
    sy: int = Field(2, description="The stride for computing dst sets.")
    seed: int = Field(
        -1,
        ge=-1,
        lt=2**32,
        description="""
Seed of the generation.
> If `-1`, then seed from `DiffusionModel` will be used.
> If `DiffusionModel.seed` is also `-1`, then random seed will be used.
""",
    )
    use_rand: bool = Field(True, description="Whether allow random perturbations.")
    merge_attn: bool = Field(True, description="Whether merge attention.")
    merge_crossattn: bool = Field(False, description="Whether merge cross attention.")
    merge_mlp: bool = Field(False, description="Whether merge mlp.")


class StyleReferenceModel(BaseModel):
    url: Optional[TImage] = Field(
        None,
        description="The url of the style image, `None` means not enabling style reference.",
    )
    style_fidelity: float = Field(
        0.5,
        description="Style fidelity, larger means reference more on the given style image.",
    )
    reference_weight: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Reference weight, similar to `control_strength`, "
            "but value > 1.0 or value < 0.0 will take no effect, "
            "so we strictly restrict it to [0.0, 1.0]."
        ),
    )


class HighresModel(BaseModel):
    fidelity: float = Field(0.3, description="Fidelity of the original latent.")
    upscale_factor: float = Field(2.0, description="Upscale factor.")
    upscale_method: str = Field("nearest-exact", description="Upscale method.")
    max_wh: int = Field(1024, description="Max width or height of the output image.")


class DiffusionModel(CallbackModel):
    use_circular: bool = Field(
        False,
        description="Whether should we use circular pattern (e.g. generate textures).",
    )
    seed: int = Field(
        -1,
        ge=-1,
        lt=2**32,
        description="""
Seed of the generation.
> If `-1`, then random seed will be used.
""",
    )
    variation_seed: int = Field(
        0,
        ge=0,
        lt=2**32,
        description="""
Seed of the variation generation.
> Only take effects when `variation_strength` is larger than 0.
""",
    )
    variation_strength: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Strength of the variation generation.",
    )
    variations: List[VariationModel] = Field(
        default_factory=lambda: [],
        description="Variation ingredients",
    )
    num_steps: int = Field(20, description="Number of sampling steps", ge=1, le=100)
    guidance_scale: float = Field(
        7.5,
        description="Guidance scale for classifier-free guidance.",
    )
    negative_prompt: str = Field(
        "",
        description="Negative prompt for classifier-free guidance.",
    )
    is_anime: bool = Field(
        False,
        description="Whether should we generate anime images or not.",
    )
    version: str = Field(
        SDVersions.v1_5,
        description="Version of the diffusion model",
    )
    sampler: SDSamplers = Field(
        SDSamplers.K_EULER,
        description="Sampler of the diffusion model",
    )
    sigmas_scheduler: Optional[SigmasScheduler] = Field(
        None,
        description="Sigmas scheduler of the k-samplers, `None` will use default.",
    )
    clip_skip: int = Field(
        -1,
        ge=-1,
        le=8,
        description="""
Number of CLIP layers that we want to skip.
> If it is set to `-1`, then `clip_skip` = 1 if `is_anime` else 0.
""",
    )
    custom_embeddings: Dict[str, Union[str, List[List[float]]]] = Field(
        {},
        description="Custom embeddings, often used in textual inversion.",
    )
    tome_info: TomeInfoModel = Field(TomeInfoModel(), description="tomesd settings.")  # type: ignore
    style_reference: StyleReferenceModel = Field(
        StyleReferenceModel(),  # type: ignore
        description="style reference settings.",
    )
    highres_info: Optional[HighresModel] = Field(None, description="Highres info.")
    lora_scales: Optional[Dict[str, float]] = Field(
        None,
        description="lora scales, key is the name, value is the weight.",
    )

    @property
    def sd_version(self) -> str:
        return self.version


class CommonSDInpaintingModel(MaxWHModel):
    keep_original: bool = Field(
        False,
        description="Whether strictly keep the original image identical in the output image.",
    )
    keep_original_num_fade_pixels: Optional[int] = Field(
        50,
        description="Number of pixels to fade the original image.",
    )
    use_raw_inpainting: bool = Field(
        False,
        description="""
Whether use the raw inpainting method.
> This is useful when you want to apply inpainting with custom SD models.
""",
    )
    use_background_guidance: bool = Field(
        False,
        description="""
Whether inject the latent of the background during the generation.
> If `use_raw_inpainting`, this will always be `True` because in this case
the latent of the background is the only information for us to inpaint.
""",
    )
    use_reference: bool = Field(
        False,
        description="Whether use the original image as reference.",
    )
    use_background_reference: bool = Field(
        False,
        description="Whether use the original image background as reference.",
    )
    reference_fidelity: float = Field(
        0.0,
        description="Fidelity of the reference image, only take effects when `use_reference` is `True`.",
    )
    inpainting_mode: InpaintingMode = Field(
        InpaintingMode.NORMAL,
        description="Inpainting mode. MASKED is preferred when the masked area is small.",
    )
    inpainting_mask_blur: Optional[int] = Field(
        None,
        description="The smoothness of the inpainting's mask, `None` means no smooth.",
    )
    inpainting_mask_padding: Optional[int] = Field(
        32,
        description="Padding of the inpainting mask under MASKED mode. If `None`, then no padding.",
    )
    inpainting_mask_binary_threshold: Optional[int] = Field(
        32,
        description="Binary threshold of the inpainting mask under MASKED mode. If `None`, then no thresholding.",
    )
    inpainting_target_wh: TNumberPair = Field(
        None,
        description="Target width and height of the images under MASKED mode.",
    )
    inpainting_padding_mode: Optional[str] = Field(None, description="Padding mode.")


class Txt2ImgDiffusionModel(DiffusionModel, MaxWHModel, TextModel):
    pass


class Img2ImgModel(MaxWHModel, ImageModel):
    pass


class Img2ImgDiffusionModel(DiffusionModel, Img2ImgModel):
    pass


# handlers


def handle_diffusion_model(
    m: DiffusionAPI,
    data: DiffusionModel,
    *,
    always_uncond: bool = True,
) -> Dict[str, Any]:
    if data.seed >= 0:
        seed = data.seed
    else:
        seed = secrets.randbelow(2**32)
    variation_seed = None
    variation_strength = None
    if data.variation_strength > 0:
        variation_seed = data.variation_seed
        variation_strength = data.variation_strength
    if data.variations is None:
        variations = None
    else:
        variations = [(v.seed, v.strength) for v in data.variations]
    m.switch_circular(data.use_circular)
    if not always_uncond and not data.negative_prompt:
        unconditional_cond = None
    else:
        unconditional_cond = [data.negative_prompt]
    clip_skip = data.clip_skip
    if clip_skip == -1:
        if data.is_anime or data.sd_version.startswith("anime"):
            clip_skip = 1
        else:
            clip_skip = 0
    # lora
    model = m.m
    if isinstance(model, StableDiffusion):
        manager = model.lora_manager
        if manager.injected:
            m.cleanup_sd_lora()
        if data.lora_scales:
            user_folder = os.path.expanduser("~")
            external_folder = os.path.join(user_folder, ".cache", "external")
            lora_folder = os.path.join(external_folder, "lora")
            for key in data.lora_scales:
                if model.lora_manager.has(key):
                    continue
                if not os.path.isdir(lora_folder):
                    raise ValueError(
                        f"'{key}' does not exist in current loaded lora "
                        f"and '{lora_folder}' does not exist either."
                    )
                for lora_file in os.listdir(lora_folder):
                    lora_name = os.path.splitext(lora_file)[0]
                    if key != lora_name:
                        continue
                    try:
                        console.log(f">> loading {key}")
                        lora_path = os.path.join(lora_folder, lora_file)
                        m.load_sd_lora(lora_name, path=lora_path)
                    except Exception as err:
                        raise ValueError(f"failed to load {key}: {err}")
            m.inject_sd_lora(*list(data.lora_scales))
            m.set_sd_lora_scales(data.lora_scales)
    # custom embeddings
    if not data.custom_embeddings:
        custom_embeddings = None
    else:
        custom_embeddings = {}
        for k, v in data.custom_embeddings.items():
            if isinstance(v, str):
                with open(v, "r") as f:
                    v = json.load(f)
            custom_embeddings[k] = v
    # return
    return dict(
        seed=seed,
        variation_seed=variation_seed,
        variation_strength=variation_strength,
        variations=variations,
        num_steps=data.num_steps,
        unconditional_guidance_scale=data.guidance_scale,
        unconditional_cond=unconditional_cond,
        sampler=data.sampler,
        sigmas_scheduler=data.sigmas_scheduler,
        verbose=OPT.verbose,
        clip_skip=clip_skip,
        custom_embeddings=custom_embeddings,
        highres_info=None
        if data.highres_info is None
        else data.highres_info.model_dump(),
    )


def handle_diffusion_inpainting_model(data: CommonSDInpaintingModel) -> Dict[str, Any]:
    return dict(
        anchor=64,
        max_wh=data.max_wh,
        keep_original=data.keep_original,
        keep_original_num_fade_pixels=data.keep_original_num_fade_pixels,
        use_raw_inpainting=data.use_raw_inpainting,
        use_background_guidance=data.use_background_guidance,
        use_reference=data.use_reference,
        use_background_reference=data.use_background_reference,
        reference_fidelity=data.reference_fidelity,
        inpainting_settings=InpaintingSettings(
            data.inpainting_mode,
            data.inpainting_mask_blur,
            data.inpainting_mask_padding,
            data.inpainting_mask_binary_threshold,
            data.inpainting_target_wh,
            data.inpainting_padding_mode,
        ),
    )


async def handle_diffusion_hooks(
    m: DiffusionAPI,
    data: DiffusionModel,
    node: IWithHttpSessionNode,
    kwargs: Dict[str, Any],
) -> None:
    # tomesd
    tome_info = data.tome_info.model_dump()
    enable_tome = tome_info.pop("enable")
    if not enable_tome:
        tome_info = None  # type: ignore
    else:
        if tome_info["seed"] == -1:
            tome_info["seed"] = kwargs.get("seed", secrets.randbelow(2**32))
    # style reference
    style_reference = data.style_reference.model_dump()
    style_image = style_reference.pop("url")
    if style_image is None:
        style_reference = None  # type: ignore
    elif not isinstance(style_image, Image.Image):
        style_image = await node.download_image(style_image)
    # setup
    m.setup_hooks(
        tome_info=tome_info,
        style_reference_image=style_image,
        style_reference_states=style_reference,
    )


class SDParameters(BaseModel):
    is_anime: bool
    sd_version: str


def get_sd_from(api_key: APIs, data: SDParameters, **kw: Any) -> ControlledDiffusionAPI:
    if not data.is_anime:
        version = data.sd_version
    else:
        version = data.sd_version if data.sd_version.startswith("anime") else "anime"
    sd: ControlledDiffusionAPI = get_api_pool().get(api_key, **kw)
    if api_key != APIs.SD_INPAINTING:
        sd.prepare_sd([version])
    elif version != SDInpaintingVersions.v1_5:
        sd.prepare_sd([version], sub_folder="inpainting", force_external=True)
    sd.switch_sd(version)
    sd.disable_control()
    return sd


def get_image_from_diffusion_output(diffusion_output: np.ndarray) -> Image.Image:
    img_arr = 0.5 * (diffusion_output + 1.0)
    img_arr = img_arr.transpose([1, 2, 0])
    img_arr = to_uint8(img_arr)
    return Image.fromarray(img_arr)
