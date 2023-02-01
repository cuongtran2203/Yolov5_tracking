from collections import deque
from venv import create
import numpy as np
import cv2
import time
from src.counting import *
from detector.YOLO_detector import Detector
from detector.func import *
p1, p2 = None, None
state = 0
# now let's initialize the list of reference point
ref_point = []
crop = False
CLASS_NAME=["bus","car","person","trailer","truck"]
def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

        
    
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
        
        self.detector=Detector()
        self.test_size=(640,640)
        
    def infer(self, img:np.ndarray):
        tic=time.time()
        outputs,bbox=self.detector.detect(img)
        cls=outputs[:,5]
        img_re=img.copy()
        if len(bbox)>0 :
            for cl,box in zip(cls,bbox) :
                box=box.cpu().detach().numpy()
                cl=cl.cpu().detach().numpy()
                box=box.astype(int)
                # print(box)
                color = get_color(abs(int(cl)))
                p1_1=(box[0],box[1])
                p2_2=(box[2],box[3])
                centerpoint=((p1_1[0]+p2_2[0])/2,(p1_1[1]+p2_2[1])/2)
                size_box=((p2_2[0]-p1_1[0]),(p2_2[1]-p1_1[1]))
                print(centerpoint)
                print(size_box)
                
                cv2.rectangle(img_re, p1_1, p2_2,color, 2, 1)
                cv2.putText(img_re,CLASS_NAME[int(cl)],(p1[0],p1[1]-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
        fps="FPS : {:.2f}".format(1/(time.time()-tic))
        print(fps)
        # cv2.putText(img_re,fps,(50,60),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,25),1)
        return img_re
    
    
if __name__ == "__main__":
    count_bus=0
    count_car=0
    count_trailer=0
    count_Truck=0
    track=Tracking_ver2()
    cap=cv2.VideoCapture("video/video5.avi")
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)
    img = np.zeros((1280,720,3), np.uint8)
    while True:
        _,frame=cap.read()
        frame=cv2.resize(frame,(1280,720))
        cv2.putText(frame,"Bus :".format(count_bus),(10,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
        cv2.putText(frame,"Car :".format(count_car),(10,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
        cv2.putText(frame,"Trailer :".format(count_trailer),(10,90),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
        cv2.putText(frame,"Truck :".format(count_Truck),(10,110),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
        if p1 is not None and p2 is not None :
            img_cropped=frame[p1[1]:p2[1],p1[0]:p2[0]]
            img_s=track.infer(img_cropped)
                # If a ROI is selected, draw it
            if state > 1:
                cv2.rectangle(img_s, p1, p2, (255, 0, 0), 10)
            cv2.rectangle(frame, (p1[0]-10,p1[1]-10),(p2[0]+10,p2[1]+10), (0, 255, 0), 3)
            frame[p1[1]:p2[1],p1[0]:p2[0]]=img_s
        cv2.imshow('frame',frame)
        if cv2.waitKey(100) & 0xff==ord("q"):
            break
            
            
        
        
    
