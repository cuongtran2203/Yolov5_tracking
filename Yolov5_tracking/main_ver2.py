from collections import deque
from venv import create
import numpy as np
import cv2
import time
from src.counting import *
from detector.YOLO_detector import Detector
from detector.func import *
from option_tracking import *
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


class Tracking_ver2():
    def __init__(self) :
        self.args= make_parser().parse_args()
        self.detector=Detector()
        self.test_size=(640,640)
    def infer(self, img:np.ndarray):
        outputs,bbox=self.detector.detect(img)
        
        
    
