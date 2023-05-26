import random
import numpy as np
import sys
import torch
from fvcore.transforms.transform import Transform, NoOpTransform
from PIL import Image
from skimage.morphology import binary_erosion, binary_dilation
from skimage.segmentation import find_boundaries
from skimage.draw import ellipse

from detectron2.data.transforms import ResizeTransform, Augmentation, ResizeShortestEdge

__all__ = [
    "ResizeTransformRGBA",
    "ResizeRGBA",
    "ResizeShortestEdgeRGBA",
    "ResizeScaleRGBA",
    "SegErosionTransform",
    "SegDilationTransform",
    "SegBulbTransform",
]


class ResizeTransformRGBA(ResizeTransform):
    def __init__(self, h, w, new_h, new_w, interp=None):
        super().__init__(h, w, new_h, new_w, interp)

    def apply_image(self, img, interp=None):
        assert img.shape[2] == 4, img.shape
        assert img.dtype == np.uint8, img.dtype
        rgb = super().apply_image(img[:, :, :3], interp)
        alpha = super().apply_image(img[:, :, 3], interp=Image.NEAREST)
        ret = np.append(rgb, alpha[..., None], axis=2)
        return ret

    def inverse(self):
        return ResizeTransformRGBA(self.new_h, self.new_w, self.h, self.w, self.interp)

    def apply_segmentation(self, segmentation):
        segmentation = super().apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class ResizeRGBA(Augmentation):
    """Resize image to a fixed target size"""

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, image):
        return ResizeTransformRGBA(
            image.shape[0], image.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdgeRGBA(ResizeShortestEdge):
    """
    Resize the image while keeping the aspect ratio unchanged.
    It attempts to scale the shorter edge to the given `short_edge_length`,
    as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    @torch.jit.unused
    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__(short_edge_length, max_size, sample_style, interp)

    @torch.jit.unused
    def get_transform(self, image):
        h, w = image.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        newh, neww = super().get_output_shape(h, w, size, self.max_size)
        return ResizeTransformRGBA(h, w, newh, neww, self.interp)


class ResizeScaleRGBA(Augmentation):
    """
    Takes target size as input and randomly scales the given target size between `min_scale`
    and `max_scale`. It then scales the input image such that it fits inside the scaled target
    box, keeping the aspect ratio constant.
    This implements the resize part of the Google's 'resize_and_crop' data augmentation:
    https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/input_utils.py#L127
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        target_height: int,
        target_width: int,
        interp: int = Image.BILINEAR,
    ):
        """
        Args:
            min_scale: minimum image scale range.
            max_scale: maximum image scale range.
            target_height: target image height.
            target_width: target image width.
            interp: image interpolation method.
        """
        super().__init__()
        self._init(locals())

    def _get_resize(self, image: np.ndarray, scale: float) -> Transform:
        input_size = image.shape[:2]

        # Compute new target size given a scale.
        target_size = (self.target_height, self.target_width)
        target_scale_size = np.multiply(target_size, scale)

        # Compute actual rescaling applied to input image and output size.
        output_scale = np.minimum(
            target_scale_size[0] / input_size[0], target_scale_size[1] / input_size[1]
        )
        output_size = np.round(np.multiply(input_size, output_scale)).astype(int)

        return ResizeTransformRGBA(
            input_size[0], input_size[1], output_size[0], output_size[1], self.interp
        )

    def get_transform(self, image: np.ndarray) -> Transform:
        random_scale = np.random.uniform(self.min_scale, self.max_scale)
        return self._get_resize(image, random_scale)


class SegErosionTransform(Transform):
    def __init__(self, max_n=5):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        assert img.dtype == np.uint8
        assert img.shape[-1] == 4, f"image shape {img.shape}"

        img = img.copy()
        seg = img[:, :, 3].astype(bool)
        times = random.randint(1, self.max_n)
        for _ in range(times):
            seg = binary_erosion(seg)
        img[:, :, 3] = seg.astype(np.uint8) * 255
        return img

    def apply_coords(self, coords):
        return coords

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation


class SegDilationTransform(SegErosionTransform):
    def __init__(self, max_n=5):
        super().__init__(max_n)

    def apply_image(self, img):
        assert img.dtype == np.uint8
        assert img.shape[-1] == 4, f"image shape {img.shape}"
        img = img.copy()
        seg = img[:, :, 3].astype(bool)
        times = random.randint(1, self.max_n)
        for _ in range(times):
            seg = binary_dilation(seg)
        img[:, :, 3] = seg.astype(np.uint8) * 255
        return img


class SegBulbTransform(SegErosionTransform):
    def __init__(self, max_n=5):
        super().__init__(max_n)

    def apply_image(self, img):
        assert img.dtype == np.uint8
        assert img.shape[-1] == 4, f"image shape {img.shape}"
        img = img.copy()
        seg = img[:, :, 3].astype(bool)
        boundaries = find_boundaries(seg)
        points_boundary = np.transpose(np.nonzero(boundaries)).tolist()
        num_points = random.randint(1, self.max_n)
        points = random.sample(points_boundary, min(num_points, len(points_boundary)))
        for point in points:
            a = random.randint(10, 30)
            b = random.randint(10, 30)
            angle = random.randint(0, 360)
            xs, ys = ellipse(point[0], point[1], a, b, rotation=np.deg2rad(angle))
            idx = (xs >= 0) & (xs < img.shape[0])
            xs, ys = xs[idx], ys[idx]
            idx = (ys >= 0) & (ys < img.shape[1])
            xs, ys = xs[idx], ys[idx]
            seg[xs, ys] = 1
        img[:, :, 3] = seg.astype(np.uint8) * 255
        return img
