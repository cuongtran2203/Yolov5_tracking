import cv2
import numpy as np
from math import sqrt

def nearest_square_number(n):
    return int(sqrt(n))+1

def square_number(n):
    sqr = sqrt(n)
    if (sqr * sqr) == n:
        return True
    else:
        return False

def process_window(image_list):
    image_length = len(image_list)
    if square_number(image_length):
        stream_tile_number = int(sqrt(image_length))
        new_width = int(1920 / stream_tile_number)
        new_height = int(1080 / stream_tile_number)
        resized_image = [cv2.resize(image, (new_width, new_height)) for image in image_list]
        output_image = cv2.vconcat([cv2.hconcat(resized_image[i:i+stream_tile_number]) for i in range(0, image_length, stream_tile_number)])
    else:
        stream_tile_number = nearest_square_number(image_length)
        expected_image_length = stream_tile_number**2
        blank_tile_number = expected_image_length - image_length
        new_width = int(1920 / stream_tile_number)
        new_height = int(1080 / stream_tile_number)
        blank_image = np.zeros((new_height, new_width, 3), np.uint8)
        image_list.extend([blank_image] * blank_tile_number)
        resized_image = [cv2.resize(image, (new_width, new_height)) for image in image_list]
        output_image = cv2.vconcat([cv2.hconcat(resized_image[i:i+stream_tile_number]) for i in range(0, expected_image_length, stream_tile_number)])
    return output_image
        
        
        
        

def read_image(path):
    image = [cv2.imread(i) for i in path]
    return image
    
if __name__ == '__main__':
    image_path = ['Stream_1.jpg', 'Stream_2.jpg', 'Stream_1.jpg', 'Stream_2.jpg', 'Stream_2.jpg']
    image = read_image(image_path)
    show_image = process_window(image)
    cv2.imshow('Image', show_image)
    cv2.waitKey(10000)
    
    
    