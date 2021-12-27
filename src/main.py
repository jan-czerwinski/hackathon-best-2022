import argparse
import datetime

from utils import (read_image, save_gray_image)
from edge_detection import EdgeDetector

def main():
    print('Script started')
    parser = argparse.ArgumentParser(description='CLI argument processing')
    parser.add_argument('--image_path', required=True, help='Path to image that is meant to be processed')
    parser.add_argument('--save_path', required=True, help='Path to save filtered image')
    args = parser.parse_args()
    
    print('Reading image')
    img = read_image(args.image_path)
    print(f'Read image with shape{img.shape}')

    print('Edge detection filtering')
    start_time = datetime.datetime.now()
    
    detector = EdgeDetector(img, gauss_size=5, sigma=1, rgb_weights=[0.2989, 0.5870, 0.1140], treshold_values=(0.05, 0.2))
    filtered_img = detector.detect()
    
    end_time = datetime.datetime.now()
    process_time = (end_time - start_time).total_seconds() * 1000
    print(f'Edge detection process time: {process_time}ms')

    print('Saving image')
    save_gray_image(args.save_path, filtered_img)
    print(f'Saved image shape{filtered_img.shape}')

    print('Script ended')

if __name__ == '__main__':
    main()
