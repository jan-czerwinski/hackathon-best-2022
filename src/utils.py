from PIL import Image
import numpy as np

def read_image(path: str) -> np.ndarray:
    img = Image.open(path)
    img = np.asarray(img)
    return img

def save_gray_image(path: str, img: np.ndarray) -> None:
    Image.fromarray(img).convert('L').save(path)

def calculate_target_size(size: int, kernel_size: int) -> int:
    num_pixels = 0

    for i in range(size):
        added = i + kernel_size
        if added <= size:
            num_pixels += 1
            
    return num_pixels

def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.shape[0]
    
    target_width_size = calculate_target_size(
        size=img.shape[0],
        kernel_size=k
    )
    target_height_size = calculate_target_size(
        size=img.shape[1],
        kernel_size=k
    )
    
    convolved_img = np.zeros(shape=(target_width_size, target_height_size))
    
    # Iterate over the rows
    for i in range(target_width_size):
        # Iterate over the columns
        for j in range(target_height_size):
            # img[i, j] = individual pixel value
            # Get the current matrix
            mat = img[i:i+k, j:j+k]
            
            # Apply the convolution - element-wise multiplication and summation of the result
            # Store the result to i-th row and j-th column of our convolved_img array
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))
            
    return convolved_img

# Functions to use later to apply padding to convolve
def get_padding_width_per_side(kernel_size: int) -> int:
    # Simple integer division
    return kernel_size // 2

def add_padding_to_image(img: np.array, padding_width: int) -> np.array:
    # Array of zeros of shape (img + padding_width)
    img_with_padding = np.zeros(shape=(
        img.shape[0] + padding_width * 2,  # Multiply with two because we need padding on all sides
        img.shape[1] + padding_width * 2
    ))
    
    # Change the inner elements
    # For example, if img.shape = (224, 224), and img_with_padding.shape = (226, 226)
    # keep the pixel wide padding on all sides, but change the other values to be the same as img
    img_with_padding[padding_width:-padding_width, padding_width:-padding_width] = img
    
    return img_with_padding
