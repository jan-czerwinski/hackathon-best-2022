from PIL import Image
import numpy as np

def read_image(path) -> np.ndarray:
    img = Image.open(path)
    img = np.asarray(img)
    return img

def save_gray_image(path, img) -> None:
    # convert('L') 
    Image.fromarray(img).convert('L').save(path)
