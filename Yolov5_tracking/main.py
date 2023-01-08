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
from src.visualize import plot_tracking
from tracker.byte_tracker import BYTETracker
from detector.YOLO_detector import Detector
from detector.func import *
from draw_ROI import *
p1, p2 = None, None
state = 0
# now let's initialize the list of reference point
ref_point = []
crop = False
  

# Called every time a mouse event happen
def on_mouse(event, x, y, flags, userdata):
    global state, p1, p2
    
    # Left click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Select first point
        if state == 0:
            p1 = (x,y)
            state += 1
        # Select second point
        elif state == 1:
            p2 = (x,y)
            state += 1
    # Right click (erase current ROI)
    if event == cv2.EVENT_LBUTTONDBLCLK:
        p1, p2 = None, None
        state = 0

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
    def infer(self,img:np.ndarray,p1,p2):
        global point_matrix
        img_info = {"id": 0}
        height, width = img.shape[:2]
        online_im=img.copy()
        # online_im=cv2.resize(online_im, (1280,720))

        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        # used to record the time at which we processed current frame
        new_frame_time = 0
        frame_id = 0
        results = []
        fps = 0
        cs=False
        # create filter class
        filter_class = [0,1,2,3,4]
        memory = {}
        angle = -1
        already_counted = deque(maxlen=50)
        prev_frame_time=time.time()
        ratio =1
        # if p1 is not None and p2 is not None:
            # img_cropped = img[p2[1]:p1[1], p2[0]:p1[0]]
        
        outputs,bbox=self.detector.detect(img)
        # print(type(outputs),outputs.shape)
        output_new=[]
        cls=outputs[:,5]
        for cl in cls :
            if cl <1  :
                cs=True
        img_info["ratio"] = ratio
        filter_class = [2,5,7]
        if outputs is not None:
            online_targets = self.tracker.update(outputs, [img_info['height'], img_info['width']], self.test_size, filter_class)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id              
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

                # origin_previous_midpoint = (previous_midpoint[0], img.shape[0] - previous_midpoint[1])
            if len(memory) > 50:
                del memory[list(memory)[0]]
            fps = 1/(time.time()-prev_frame_time) 
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1)
            # cv2.putText(online_im, "FPS: {:.2f}".format(fps),(50,100),cv2.FONT_HERSHEY_TRIPLEX,2,(255,0,0),1) 
            if cs :
                cv2.putText(online_im, "Person ",(450,100),cv2.FONT_HERSHEY_TRIPLEX,2,(0,0,255),1)
            else :
                cv2.putText(online_im, " No Person ",(450,100),cv2.FONT_HERSHEY_TRIPLEX,2,(0,0,255),1)
            
        return online_im
        # else :
        #     return online_im
if __name__ == '__main__':

    cap=cv2.VideoCapture("video/video5.avi")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img = np.zeros((1280,720,3), np.uint8)
    track=Tracking()
    count=0
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)
    while cap.isOpened():
        _,frame=cap.read()
        count+=1
        if count>length:
            break
        frame=cv2.resize(frame,(1280,720))
        img=frame.copy()
        img_s=frame.copy()
        cv2.setMouseCallback('frame', on_mouse)
        # Cropping image
        if p1 is not None and p2 is not None :
            img_cropped=frame[p1[1]:p2[1],p1[0]:p2[0]]
            img_s=track.infer(img_cropped,p1,p2)
                # If a ROI is selected, draw it
            if state > 1:
                cv2.rectangle(img_s, p1, p2, (255, 0, 0), 10)
            cv2.rectangle(frame, (p1[0]-10,p1[1]-10),(p2[0]+10,p2[1]+10), (0, 255, 0), 3)
            frame[p1[1]:p2[1],p1[0]:p2[0]]=img_s

        cv2.imshow('frame',frame)
        # print(p1,p2)

        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        
        
        
        
