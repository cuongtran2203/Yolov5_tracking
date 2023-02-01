from collections import deque
from venv import create
import numpy as np
import cv2
import time
from src.counting import *
from detector.YOLO_detector import Detector
from detector.func import *
from option_tracking import *
from lane_line_detector import *
p1, p2 = None, None
state = 0
# now let's initialize the list of reference point
ref_point = []
crop = False
CLASS_NAME=["bus","car","person","trailer","truck"]
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
        tic=time.time()
        mask=find_lane_line(img)
        outputs,bbox=self.detector.detect(img)
        cls=outputs[:,5]
        print(cls)
        img_re=img.copy()
        if len(bbox)>0 :
            for cl,box in zip(cls,bbox) :
                box=box.cpu().detach().numpy()
                box=box.astype(int)
                box=tlbr_to_tlwh(box)
                # print(box)
                img_re=tracking(self.args,img,box,cl)
        fps="FPS : {:.2f}".format(1/(time.time()-tic))
        # print(fps)
        cv2.putText(img_re,fps,(50,60),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,25),1)
        return img_re
    
    
if __name__ == "__main__":
    track=Tracking_ver2()
    cap=cv2.VideoCapture("video/video5.avi")
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)
    img = np.zeros((1280,720,3), np.uint8)
    
    while True:
        _,frame=cap.read()
        frame=cv2.resize(frame,(1280,720))
        # if p1 is not None and p2 is not None :
        #     img_cropped=frame[p1[1]:p2[1],p1[0]:p2[0]]
        #     img_s=track.infer(img_cropped)
        #         # If a ROI is selected, draw it
        #     if state > 1:
        #         cv2.rectangle(img_s, p1, p2, (255, 0, 0), 10)
        #     cv2.rectangle(frame, (p1[0]-10,p1[1]-10),(p2[0]+10,p2[1]+10), (0, 255, 0), 3)
        #     frame[p1[1]:p2[1],p1[0]:p2[0]]=img_s
        track.infer(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(100) & 0xff==ord("q"):
            break
            
            
        
        
    
