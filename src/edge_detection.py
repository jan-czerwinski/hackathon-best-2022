import numpy as np
import datetime
from PIL import Image
import argparse
from scipy.ndimage.filters import convolve

class EdgeDetector:

    def __init__(self, img, gauss_size, sigma, rgb_weights) -> None:
        self.img = img
        self.size = gauss_size
        self.sigma = sigma
        self.rgb_weights = rgb_weights

    def gaussian_kernel(self, size, sigma):
        half_size = int(size) // 2
        x, y = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * (1 / (2.0 * np.pi * sigma**2))
        return g

    def intensity_gradient(self, img):
        sobel_filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
        sobel_fiter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        # TODO: create convolve function
        gx = convolve(img, sobel_filter_x)
        # TODO: create convolve function
        gy = convolve(img, sobel_fiter_y)

        # g = sqrt(gx^2 + gy^2)
        g = np.hypot(gx, gy)
        # normalization
        g = g / g.max() * 255
        
        teta = np.arctan2(gx, gy)

        return (g, teta)

    # img as ndarray with shape (width, height, channels)
    def edge_detection(self, img) -> np.ndarray:

        # applying grayscale to img
        # filtered_image should be shape (width, height)
        filtered_image = np.dot(img[..., :3], self.rgb_weights)

        # apllying gaussian kernel with shape (5,5)
        # TODO: create convolve function
        filtered_image = convolve(filtered_image, self.gaussian_kernel(size=self.size, sigma=self.sigma))

        # finding intensity gradient
        filtered_image, teta = self.intensity_gradient(filtered_image)

        return filtered_image

    def detect(self) -> np.ndarray:
        return self.edge_detection(self.img)
