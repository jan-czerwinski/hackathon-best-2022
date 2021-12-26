import numpy as np
# from utils import convolve
from scipy.ndimage.filters import convolve

class EdgeDetector:

    def __init__(self, img: np.ndarray, gauss_size: int, sigma: float, rgb_weights: list) -> None:
        self.img = img
        self.size = gauss_size
        self.sigma = sigma
        self.rgb_weights = rgb_weights

    def gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        half_size = int(size) // 2
        x, y = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * (1 / (2.0 * np.pi * sigma**2))
        return g

    def intensity_gradient(self, img: np.ndarray):
        sobel_filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
        sobel_fiter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        gx = convolve(img, sobel_filter_x)
        gy = convolve(img, sobel_fiter_y)

        # g = sqrt(gx^2 + gy^2)
        g = np.hypot(gx, gy)
        # normalization
        g = g / g.max() * 255

        teta = np.arctan2(gx, gy)

        return g, teta

    def non_max_suppression(self, src_img: np.ndarray, threshold: np.ndarray):
        src_width, src_height = src_img.shape

        out_img = np.zeros((src_width, src_height), dtype=np.int32)
        angle = threshold * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, src_width - 1):
            for j in range(1, src_height - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = src_img[i, j + 1]
                        r = src_img[i, j - 1]
                    # angle 45
                    elif 22.5 <= angle[i, j] < 67.5:
                        q = src_img[i + 1, j - 1]
                        r = src_img[i - 1, j + 1]
                    # angle 90
                    elif 67.5 <= angle[i, j] < 112.5:
                        q = src_img[i + 1, j]
                        r = src_img[i - 1, j]
                    # angle 135
                    elif 112.5 <= angle[i, j] < 157.5:
                        q = src_img[i - 1, j - 1]
                        r = src_img[i + 1, j + 1]

                    if (src_img[i, j] >= q) and (src_img[i, j] >= r):
                        out_img[i, j] = src_img[i, j]
                    else:
                        out_img[i, j] = 0

                except IndexError as e:
                    pass

        return out_img

    # img as ndarray with shape (height, width, channels)
    def edge_detection(self, img: np.ndarray) -> np.ndarray:

        # applying grayscale to img
        # filtered_image should be shape (height, width)
        filtered_image = np.dot(img[..., :3], self.rgb_weights)

        # apllying gaussian kernel with shape (5,5)
        filtered_image = convolve(filtered_image, self.gaussian_kernel(size=self.size, sigma=self.sigma))

        # finding intensity gradient
        filtered_image, teta = self.intensity_gradient(filtered_image)

        filtered_image = self.non_max_suppression(filtered_image, teta)

        return filtered_image

    def detect(self) -> np.ndarray:
        return self.edge_detection(self.img)
