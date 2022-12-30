from ast import arg
from collections import deque
import sys
from venv import create
import torch
import numpy as np
import cv2
import time
from src.counting import *
from src.coco_name import COCO_CLASSES
from src.ultils import postprocess
from src.visualize import vis
from src.visualize import plot_tracking
from tracker.byte_tracker import BYTETracker
from detector.YOLO_detector import Detector
COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)
class Args():
    def __init__(self) -> None:
        self.track_thresh = 0.4
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False
        self.tsize = None
        self.exp_file = None
class Tracking():
    def __init__(self) -> None:
        super(Tracking,self).__init__()
        args=Args()
        self.tracker = BYTETracker(args, frame_rate=22)
        self.detector = Detector()
        self.test_size=(640,640)
    def infer(self,img:np.ndarray):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        # used to record the time at which we processed current frame
        new_frame_time = 0
        frame_id = 0
        results = []
        fps = 0
        # create filter class
        filter_class = [2]
        # init variable for counting object
        memory = {}
        angle = -1
        already_counted = deque(maxlen=50)
        prev_frame_time=time.time()
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        outputs=self.detector.detect(img)
        img_info["ratio"] = ratio
        filter_class = [2]
        if outputs is not None:
            online_targets = self.tracker.update(outputs, [img_info['height'], img_info['width']], self.test_size, filter_class)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                print(tid)                
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                results.append(f"{frame_id}, {tid}, {tlwh[0]:.2f}, {tlwh[1]:.2f}, {tlwh[2]:.2f}, {tlwh[3]:.2f},{ t.score:.2f}, -1, -1, -1\n")
                # couting
                # get midpoint from bbox
                midpoint = tlbr_midpoint(tlwh)
                origin_midpoint = (midpoint[0], img.shape[0] - midpoint[1])  # get midpoint respective to bottom-left

                if tid not in memory:
                    memory[tid] = deque(maxlen=2)

                memory[tid].append(midpoint)
                previous_midpoint = memory[tid][0]

                origin_previous_midpoint = (previous_midpoint[0], img.shape[0] - previous_midpoint[1])

            if len(memory) > 50:
                del memory[list(memory)[0]]
            fps = 1/(time.time()-prev_frame_time) 
                  
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,fps=fps)
            cv2.putText(online_im, "FPS: {:.2f}".format(fps),(50,100),cv2.FONT_HERSHEY_TRIPLEX,2,(0,0,255),1) 
        else:
            online_im = img_info['raw_img']

        # font which we will be using to display FPS
        font = cv2.FONT_HERSHEY_SIMPLEX
        # time when we finish srcing for this frame
        new_frame_time = time.time()
    
        # Calculating the fps
       
        return online_im
if __name__ == '__main__':
    cap=cv2.VideoCapture("video/video5.avi")
    track=Tracking()
    while cap.isOpened():
        _,frame=cap.read()
        img=track.infer(frame)
        img=cv2.resize(img,(1280,720))

        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        
        
