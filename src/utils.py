from PIL import Image
import numpy as np


def read_image(path: str) -> np.ndarray:
    img = Image.open(path)
    img = np.asarray(img)
    return img


def save_gray_image(path: str, img: np.ndarray) -> None:
    Image.fromarray(img).convert('L').save(path)


def pad_array(img: np.ndarray, pad: int) -> np.ndarray:
    """3D convolution by sub-matrix summing.
    Args:
        img (ndarray): Array to be padded
        pad (int): Amount of padding on each side
    Returns:
        result (ndarray): Padded array
    """
    if pad == 0:
        return img
    var_pad = np.zeros(tuple(pad * 2 + np.array(img.shape[:2])) + img.shape[2:])
    var_pad[pad, pad] = img
    return var_pad


def convolve(img: np.ndarray, kernel: np.ndarray, pad: int = 0) -> np.ndarray:
    """3D convolution by sub-matrix summing.
    Args:
        img (ndarray): 2d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d kernel to convolve
        pad (int): number of columns/rows to pad at edges.
    Returns:
        result (ndarray): convolution result.
    """
    ny, nx = img.shape[:2]
    ky, kx = kernel.shape[:2]
    result = 0
    if pad > 0:
        var_pad = pad_array(img, pad)
    else:
        var_pad = img
    for ii in range(ky * kx):
        yi, xi = divmod(ii, kx)
        slabii = var_pad[yi:2 * pad + ny - ky + yi + 1:1, xi:2 * pad + nx - kx + xi + 1:1, ...] * kernel[yi, xi]
        result += slabii
    return result
