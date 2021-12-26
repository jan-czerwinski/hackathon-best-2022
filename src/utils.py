from PIL import Image
import numpy as np

def read_image(path: str) -> np.ndarray:
    img = Image.open(path)
    img = np.asarray(img)
    return img

def save_gray_image(path: str, img: np.ndarray) -> None:
    Image.fromarray(img).convert('L').save(path)

# convolve function without padding
# causes image to reduce shape
# TODO: add padding / change convolve method
def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.shape[0]
    y,x = img.shape
    y = y - k + 1
    x = x - k + 1
    convolved_image = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            convolved_image[i][j] = np.sum(img[i:i+k, j:j+k] * kernel)
    return convolved_image
