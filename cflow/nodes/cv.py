# Computer Vision Nodes

import cv2

import numpy as np

from PIL import Image
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from pydantic import Field
from pydantic import BaseModel
from PIL.Image import Resampling as PILResampling
from PIL.ImageFilter import GaussianBlur
from cftool.cv import to_rgb
from cftool.cv import to_uint8
from cftool.cv import to_base64
from cftool.cv import ImageBox
from cftool.misc import safe_execute
from cftool.misc import shallow_copy_dict
from cftool.geometry import Matrix2D

from .schema import TImage
from .schema import DocEnum
from .schema import DocModel
from .schema import ImageModel
from .schema import IImageNode
from .schema import IWithImageNode
from ..core import Node
from ..core import Schema


# common


class WHModel(DocModel):
    w: int = Field(..., description="Width of the output image.")
    h: int = Field(..., description="Height of the output image")


class LTRBModel(DocModel):
    lt_rb: Tuple[int, int, int, int] = Field(
        ...,
        description="The left-top and right-bottom points.",
    )


class ResizeMode(str, DocEnum):
    FILL = "fill"
    FIT = "fit"
    COVER = "cover"


class Resampling(str, DocEnum):
    NEAREST = "nearest"
    BOX = "box"
    BILINEAR = "bilinear"
    HAMMING = "hamming"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


class ResamplingModel(DocModel):
    resampling: Resampling = Field(Resampling.BILINEAR, description="The resampling.")


class BaseAffineModel(DocModel):
    a: float = Field(..., description="`a` of the affine matrix")
    b: float = Field(..., description="`b` of the affine matrix")
    c: float = Field(..., description="`c` of the affine matrix")
    d: float = Field(..., description="`d` of the affine matrix")
    e: float = Field(..., description="`e` of the affine matrix")
    f: float = Field(..., description="`f` of the affine matrix")
    wh_limit: int = Field(
        16384,
        description="maximum width or height of the output image",
    )


def erode(array: np.ndarray, n_iter: int, kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(array, kernel, iterations=n_iter)


def resize(image: Image.Image, w: int, h: int, resampling: Resampling) -> Image.Image:
    if resampling == Resampling.NEAREST:
        r = PILResampling.NEAREST
    elif resampling == Resampling.BOX:
        r = PILResampling.BOX
    elif resampling == Resampling.BILINEAR:
        r = PILResampling.BILINEAR
    elif resampling == Resampling.HAMMING:
        r = PILResampling.HAMMING
    elif resampling == Resampling.BICUBIC:
        r = PILResampling.BICUBIC
    else:
        r = PILResampling.LANCZOS
    return image.resize((w, h), r)


def affine(
    image: Image.Image,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    w: int,
    h: int,
    resampling: Resampling,
    wh_limit: int,
) -> np.ndarray:
    matrix2d = Matrix2D(a=a, b=b, c=c, d=d, e=e, f=f)
    properties = matrix2d.decompose()
    iw, ih = image.size
    nw = max(round(iw * abs(properties.w)), 1)
    nh = max(round(ih * abs(properties.h)), 1)
    if nw > wh_limit or nh > wh_limit:
        raise ValueError(f"image size ({nw}, {nh}) exceeds wh_limit ({wh_limit})")
    array = np.array(resize(image, nw, nh, resampling))
    properties.w = 1
    properties.h = 1 if properties.h > 0 else -1
    matrix2d = Matrix2D.from_properties(properties)
    return cv2.warpAffine(array, matrix2d.matrix, [w, h])


def paste(
    fg: Image.Image,
    bg: Image.Image,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    force_rgb: bool,
    resampling: Resampling,
    wh_limit: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fg.mode != "RGBA":
        fg = fg.convert("RGBA")
    w, h = bg.size
    affined = affine(fg, a, b, c, d, e, f, w, h, resampling, wh_limit)
    affined = affined.astype(np.float32) / 255.0
    rgb = affined[..., :3]
    mask = affined[..., -1:]
    if force_rgb:
        bg = to_rgb(bg)
    bg_array = np.array(bg).astype(np.float32) / 255.0
    fg_array = rgb if bg_array.shape[2] == 3 else affined
    pasted = fg_array * mask + bg_array * (1.0 - mask)
    return to_uint8(rgb), to_uint8(mask[..., 0]), to_uint8(pasted)


def get_contrast_bg(rgba_image: Image.Image) -> int:
    rgba = np.array(rgba_image)
    rgb = rgba[..., :3]
    alpha = rgba[..., -1]
    target_mask = alpha >= 10
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    lightness = hls[..., 1].astype(np.float32) / 255.0
    target_lightness = lightness[target_mask]
    mean = target_lightness.mean().item()
    std = target_lightness.std().item()
    if 0.45 <= mean <= 0.55 and std >= 0.25:
        return 127
    if mean <= 0.2 or 0.8 <= mean:
        return 127
    return 0 if mean >= 0.5 else 255


# nodes


class BinaryOpInput(ImageModel):
    url2: TImage = Field(..., description="The second image to process.")
    op: str = Field(..., description="The (numpy) operation.")


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


class BinarizeInput(ImageModel):
    threshold: int = Field(127, description="The threshold of the binarization.")


@Node.register("cv.binarize")
class BinarizeNode(IImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.input_model = BinarizeInput
        schema.description = "Binarize an image."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        array = np.array(await self.get_image_from("url"))
        threshold = self.data["threshold"]
        binarized = np.where(array > threshold, 255, 0).astype(np.uint8)
        image = Image.fromarray(binarized)
        return {"image": image}


class BlurInput(ImageModel):
    radius: int = Field(2, description="Size of the blurring kernel.")


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


class ErodeAlphaInput(ImageModel):
    n_iter: int = Field(1, description="Number of iterations.")
    kernel_size: int = Field(3, description="Size of the kernel.")
    threshold: int = Field(0, description="Threshold of the alpha channel.")
    padding: int = Field(8, description="Padding for the image.")


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


class ResizeInput(ResamplingModel, WHModel, ImageModel):
    mode: ResizeMode = Field(ResizeMode.FILL, description="The Resize mode.")


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


class AffineInput(ResamplingModel, WHModel, BaseAffineModel, ImageModel):
    force_rgba: bool = Field(
        False,
        description="Whether to force the output to be RGBA.",
    )


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


class GetMaskInput(ImageModel):
    get_inverse: bool = Field(False, description="Whether get the inverse mask.")
    binarize_threshold: Optional[int] = Field(
        None,
        ge=0,
        le=255,
        description="If not `None`, will binarize the mask with this value.",
    )


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


class FillBGInput(ImageModel):
    bg: Optional[Union[int, Tuple[int, int, int]]] = Field(
        None,
        description="""Target background color.
> If not specified, `get_contrast_bg` will be used to calculate the `bg`.
""",
    )


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


class GetSizeOutput(BaseModel):
    size: Tuple[int, int] = Field(..., description="The size of the image.")


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


class ModifyBoxInput(LTRBModel):
    w: Optional[int] = Field(None, description="The width of the image.")
    h: Optional[int] = Field(None, description="The height of the image.")
    padding: int = Field(0, description="The padding size.")
    to_square: bool = Field(False, description="Turn the box into a square box.")


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


class GenerateMasksInput(BaseModel):
    w: int = Field(..., description="The width of the canvas.")
    h: int = Field(..., description="The height of the canvas.")
    lt_rb_list: List[Tuple[int, int, int, int]] = Field(..., description="The boxes.")
    merge: bool = Field(False, description="Whether merge the masks.")


class GenerateMaskAPIOutput(BaseModel):
    masks: List[str] = Field(..., description="The base64 encoded masks.")


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


class CropImageInput(LTRBModel, ImageModel):
    pass


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


class PasteModel(BaseModel):
    bg_url: TImage = Field(..., description="The background image.")
    force_rgb: bool = Field(False, description="Whether to force the output to be RGB.")


class PasteInput(ResamplingModel, BaseAffineModel, PasteModel, ImageModel):
    pass


class PasteAPIOutput(BaseModel):
    rgb: str = Field(
        ...,
        description="The base64 encoded RGB image.\n"
        "> This is the affined `fg` which does not contain the background.",
    )
    mask: str = Field(
        ...,
        description="The base64 encoded mask image.\n"
        "> This is the affined `fg`'s alpha channel.",
    )
    pasted: str = Field(..., description="The base64 encoded pasted image.")


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


__all__ = [
    "ResizeMode",
    "Resampling",
    "WHModel",
    "LTRBModel",
    "ResizeMode",
    "ResamplingModel",
    "BaseAffineModel",
    "BinaryOpNode",
    "BinarizeNode",
    "BlurNode",
    "InverseNode",
    "GrayscaleNode",
    "ErodeAlphaNode",
    "ResizeNode",
    "AffineNode",
    "GetMaskNode",
    "FillBGNode",
    "GetSizeNode",
    "ModifyBoxNode",
    "GenerateMasksNode",
    "CropImageNode",
]
