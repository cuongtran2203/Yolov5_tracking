from ast import arg
from collections import deque
import sys
from venv import create
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
from lane_line_detector import *
import time
p1, p2,p3,p4 = None, None,None,None
state = 0
# now let's initialize the list of reference point
ref_point = []
crop = False
CLASS_NAME=["bus","car","person","trailer","truck"]
def get_area_detect(img, points):
    # points = points.reshape((-1, 1, 2))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dts = cv2.bitwise_and(img, img, mask=mask)
    return dts
def estimate_velocity(len_contours=None,A=[None,None],B=[None,None],delta_t=None) :
    y_min=min(p1[1],p2[1])
    y_max=max(p3[1],p4[1])
    distance_constant=len_contours+3*(len_contours-1)
    meters_per_pixel=distance_constant/(y_max-y_min)
    distance_in_pixel=abs(A[1]-B[1])
    distance_in_meter_zone=meters_per_pixel*distance_in_pixel
    speed=distance_in_meter_zone/(delta_t)
    return speed*3.6
    

# Called every time a mouse event happen
def on_mouse(event, x, y, flags, userdata):
    global state, p1, p2,p3,p4
    
    # Left click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Select first point
        if state == 0:
            p1 = [x,y]
            print(p1)
            state += 1
        # Select second point
        elif state == 1:
            p2 = [x,y]
            print(p2)
            state += 1
          # Select second point
        elif state == 2:
            p3 = [x,y]
            print(p3)
            state += 1
          # Select second point
        elif state == 3:
            p4 = [x,y]
            print(p4)
            state += 1
    # Right click (erase current ROI)
    if event == cv2.EVENT_LBUTTONDBLCLK:
        p1, p2 = None, None
        state = 0
M={"bus":[],"car":[],"trailer":[],"truck":[],"person":[]}

COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)
class Args():
    def __init__(self) -> None:
        self.track_thresh = 0.4
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = True
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
        online_im=img.copy()
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        frame_id = 0

        cs=False
        # create filter class
        filter_class = [0,1,3,4]
        ratio =1       
        outputs,bbox=self.detector.detect(img)
        cls=outputs[:,5]
        img_info["ratio"] = ratio
        filter_class = [0,1,3,4]
        
        if outputs is not None:
            online_targets = self.tracker.update(outputs, [img_info['height'], img_info['width']], self.test_size, filter_class)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            current_dict_info={}
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id              
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                midpoint = tlbr_midpoint(tlwh)
                origin_midpoint = (midpoint[0], img.shape[0] - midpoint[1]) # get midpoint respective to bottom-left
                if check%3==0:
                    info_list_id=[origin_midpoint,time.time()]#save info ids [ (cx,cy),time]
                    current_dict_info.update({str(tid):info_list_id})
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1)
            for cl, id,box,speed in zip(cls,online_ids,bbox,speed_list) :
                if str(id) not in M[CLASS_NAME[int(cl)]]:
                    M[CLASS_NAME[int(cl)]].append(str(id))
                if int(cl)==2 :
                    cs=True
                cv2.putText(online_im,"{:.2f} km/h".format(speed),(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,24,25),2)
                
            speed_list.clear()
            count_car=len(M["car"])
            count_bus=len(M["bus"])
            count_trailer=len(M["trailer"])
            count_truck=len(M["truck"])
            
        speed_list.clear()
        return online_im,count_car,count_bus,count_trailer,count_truck,cs,current_dict_info,bbox

if __name__ == '__main__':
    prev_dict_id ={}
    cap=cv2.VideoCapture("video/video5.avi")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img = np.zeros((1280,720,3), np.uint8)
    track=Tracking()
    count=0
    check=0
    speed_list=[]
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)
    while True:
        _,frame=cap.read()
        c3_new=[]
        count+=1
        if count>length:
            break
        frame=cv2.resize(frame,(1280,720))
        img=frame.copy()
        img_s=frame.copy()
        cv2.setMouseCallback('frame', on_mouse)
        # Cropping image
        if p1 is not None and p2 is not None and p3 is not None and p4 is not None :
            pts=np.array([p1,p2,p3,p4],np.int32)
            img_croped=get_area_detect(img,pts)
            img_copy=img_croped.copy()
            img_s,count_car,count_bus,count_trailer,count_truck,cs,dict_info,bbox=track.infer(img_croped)
            for box in bbox :
                box=list(map(int,box))
                img_vehicle=np.zeros([box[3]-box[1],box[2]-box[0],3],dtype=np.uint8,order='C')
                img_copy[box[1]:box[3],box[0]:box[2]]=img_vehicle
            mask=find_lane_line(img_copy)
            center=[[abs(int((p1[0]+p2[0])/2)),abs(int((p1[1]+p2[1])/2))],[abs(int((p3[0]+p4[0])/2)),abs(int((p3[1]+p4[1])/2))]]
            contours = find_contour(mask)
            for c in contours :
                if len(c)>20 and len(c)<100:
                    dist=calculate_distance(c[0][0],np.array(center))
                    if dist <50 :
                        c3_new.append(c)
                        cv2.drawContours(img_s,c,-1,(0,0,255),3)
            cv2.putText(img,"Bus : {}".format(count_bus),(10,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            cv2.putText(img,"Car : {}".format(count_car),(10,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            cv2.putText(img,"Trailer : {}".format(count_trailer),(10,90),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            cv2.putText(img,"Truck : {}".format(count_truck),(10,110),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            if cs :
                cv2.putText(img,"Person : Found ",(10,130),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            else :
                cv2.putText(img,"Person : Not Found ",(10,130),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            # print("Previous dict : ",prev_dict_id)
            if len(dict_info)>0 and len(prev_dict_id)>0:
                for key in dict_info.keys():
                    if key in prev_dict_id.keys():
                        A=prev_dict_id[key][0]
                        B=dict_info[key][0]
                        deta_t=abs(prev_dict_id[key][1]-dict_info[key][1])
                        speed=estimate_velocity(len(c3_new),A,B,deta_t)
                        # print("Speed Vehicle: ",speed)
                        speed_list.append(speed)
            cv2.imshow("sss",img_s)
            check+=1
            if len(dict_info)>0:
                prev_dict_id=dict_info.copy()
        cv2.imshow('frame',img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        
        
        
