import cv2 
import torch 
import numpy as np
from detector.YOLO_detector import Detector
import multiprocessing
from multiprocessing import Pool,Process,Manager,Queue
from multiprocessing.pool import ThreadPool
import time
from concurrent.futures import ProcessPoolExecutor
from math import sqrt
from threading import Thread
from numba import njit ,prange
# @njit(parallel=True)
# def read_video(video_list):
#     frame_list = []
#     for vid in video_list:
#         ret,frame=vid.read()
#         if ret:
#             frame_list.append(frame)
#     return frame_list

# intialize global variables for the pool processes:
class Video_thread(Process):
    def __init__(self,video_file):
        super().__init__()
        self.video_file = video_file
    def run(self):
        cap=cv2.VideoCapture(self.video_file)
        while True:
            ret,frame=cap.read()
            tasks_to_accomplish.put(frame)
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
def process_excute(frame):
    output=model.detect(frame)
    # tasks_that_are_done.put(output)
    return output
def show(frame_list):
        img=process_window(frame_list)

    # time.sleep(1)
if __name__ == "__main__":
    model=Detector()
    detection_buffer = Queue()
    process=[]
    tasks_to_accomplish = Queue(maxsize=1000000)
    tasks_that_are_done = Queue(maxsize=100000)
    check=0
    v1=cv2.VideoCapture("./video/video1.avi")
    v2=cv2.VideoCapture("./video/video2.avi")
    v3=cv2.VideoCapture("./video/video3.avi")
    v4=cv2.VideoCapture("./video/video4.avi")
    v5=cv2.VideoCapture("./video/video5.avi")
    v6=cv2.VideoCapture("./video/video6.avi")
    v7=cv2.VideoCapture("./video/video7.avi")
    v8=cv2.VideoCapture("./video/video8.avi")
    v9=cv2.VideoCapture("./video/video9.avi")
    v10=cv2.VideoCapture("./video/video10.avi")
    v11=cv2.VideoCapture("./video/video11.avi")
    v12=cv2.VideoCapture("./video/video13.avi")
    v13=cv2.VideoCapture("./video/video14.avi")
    v14=cv2.VideoCapture("./video/video14.avi")
    v15=cv2.VideoCapture("./video/video0.avi")
    v16=cv2.VideoCapture("./video/video15.avi")
    video_list=[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,v10,v11,v12, v13, v14, v15, v16]
    while True :
        ret1,s1=v1.read()
        ret2,s2=v2.read()
        ret3,s3=v3.read()
        ret4,s4=v4.read()
        ret5,s5=v5.read()
        ret6,s6=v6.read()
        ret7,s7=v7.read()
        ret8,s8=v8.read()
        ret9,s9=v9.read()
        ret10,s10=v10.read()
        ret11,s11=v11.read()
        ret12,s12=v12.read()
        ret13,s13=v13.read()
        ret14,s14=v14.read()
        ret15,s15=v15.read()
        ret16,s16=v16.read()
        frame_list=[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s1,s2,s3,s4,s5,s6,s7,s8,s9]
        img=process_window(frame_list)
        cv2.imshow("img",img)
        cv2.waitKey(25)
        frame_lis=[]
        if check %30 ==0:
            with ProcessPoolExecutor(max_workers=7) as executor:
                result= executor.map(process_excute,frame_list)
        check+=1
        

       


        

            
            
    
    