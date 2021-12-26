from PIL import Image
import numpy as np


def read_image(path: str) -> np.ndarray:
    img = Image.open(path)
    img = np.asarray(img)
    return img


def save_gray_image(path: str, img: np.ndarray) -> None:
    Image.fromarray(img).convert('L').save(path)


def pad_array(img, pad1: int, pad2: int = None) -> np.ndarray:
    """Pad array with 0s
    Args:
        img (ndarray): 2d or 3d ndarray. Padding is done on the first 2 dimensions.
        pad1 (int): number of columns/rows to pad at left/top edges.
    Keyword Args:
        pad2 (int): number of columns/rows to pad at right/bottom edges.
            If None, same as <pad1>.
    Returns:
        var_pad (ndarray): 2d or 3d ndarray with 0s padded along the first 2
            dimensions.
    """
    if pad2 is None:
        pad2 = pad1
    if pad1 + pad2 == 0:
        return img
    var_pad = np.zeros(tuple(pad1 + pad2 + np.array(img.shape[:2])) + img.shape[2:])
    var_pad[pad1:-pad2, pad1:-pad2] = img
    return var_pad


def pick_strided(img: np.ndarray, stride: int) -> np.ndarray:
    """Pick sub-array by stride
    Args:
        img (ndarray): 2d or 3d ndarray.
        stride (int): stride/step along the 1st 2 dimensions to pick
            elements from <var>.
    Returns:
        result (ndarray): 2d or 3d ndarray picked at <stride> from <var>.
    """
    if stride < 0:
        raise Exception("<stride> should be >=1.")
    if stride == 1:
        result = img
    else:
        result = img[::stride, ::stride, ...]
    return result


# convolve function without padding
# causes image to reduce shape
# TODO: add padding / change convolve method
def convolve(img: np.ndarray, kernel: np.ndarray, stride=1, pad=0) -> np.ndarray:
    """3D convolution by sub-matrix summing.
    Args:
        img (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
        pad (int): number of columns/rows to pad at edges.
    Returns:
        result (ndarray): convolution result.
    """
    var_ndim = np.ndim(img)
    ny, nx = img.shape[:2]
    ky, kx = kernel.shape[:2]
    result = 0
    if pad > 0:
        var_pad = pad_array(img, pad, pad)
    else:
        var_pad = img
    for ii in range(ky * kx):
        yi, xi = divmod(ii, kx)
        slabii = var_pad[yi:2 * pad + ny - ky + yi + 1:1,
                 xi:2 * pad + nx - kx + xi + 1:1, ...] * kernel[yi, xi]
        if var_ndim == 3:
            slabii = slabii.sum(axis=-1)
        result += slabii
    if stride > 1:
        result = pick_strided(img, stride)
    return result
