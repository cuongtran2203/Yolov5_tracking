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
def estimate_velocity(frame:np.ndarray, len_contours=None,A=[None,None],B=[None,None],delta_t=None) :
    distance_constant=len(len_contours)+3*(len_contours-1)
    meters_per_pixel=distance_constant/frame.shape[0]
    distance_in_pixel=abs(A[1]-B[1])
    distance_in_meter_zone=meters_per_pixel*distance_in_pixel
    speed=distance_in_meter_zone/delta_t
    return speed
    
def count(founded_classes, im0):

  for i, (k,v) in enumerate(founded_classes.items()):
    cnt_str = str(k) + ":" + str(v)
    height, width, _ = im0.shape
    #cv2.line(im0, (20,65+ (i*40)), (127,65+ (i*40)), [85,45,255], 30)
    if str(k) == 'car':
        i = 0
        cv2.rectangle(im0, (width - 190, 45 + (i*40)), (width, 95 + (i*40)), [85, 45, 255], -1,  cv2.LINE_AA)
        cv2.putText(im0, cnt_str, (width - 190, 75 + (i*40)), 0, 1, [255, 0, 0], thickness = 2, lineType = cv2.LINE_AA)
    elif str(k) == 'bus':
        i = 1
        cv2.rectangle(im0, (width - 190, 45 + (i*40)), (width, 95 + (i*40)), [85, 45, 255], -1,  cv2.LINE_AA)
        cv2.putText(im0, cnt_str, (width - 190, 75 + (i*40)), 0, 1, [255, 0, 0], thickness = 2, lineType = cv2.LINE_AA)
    elif str(k) == 'truck':
        i = 2
        cv2.rectangle(im0, (width - 190, 45 + (i*40)), (width, 95 + (i*40)), [85, 45, 255], -1,  cv2.LINE_AA)
        cv2.putText(im0, cnt_str, (width - 190, 75 + (i*40)), 0, 1, [255, 0, 0], thickness = 2, lineType = cv2.LINE_AA)
    elif str(k) == 'trailer':
        i = 3
        cv2.rectangle(im0, (width - 190, 45 + (i*40)), (width, 95 + (i*40)), [85, 45, 255], -1,  cv2.LINE_AA)
        cv2.putText(im0, cnt_str, (width - 190, 75 + (i*40)), 0, 1, [255, 0, 0], thickness = 2, lineType = cv2.LINE_AA)
    # elif str(k) == 'person':
    #     i = 3
    #     cv2.rectangle(im0, (width - 190, 45 + (i*40)), (width, 95 + (i*40)), [85, 45, 255], -1,  cv2.LINE_AA)
    #     cv2.putText(im0, cnt_str, (width - 190, 75 + (i*40)), 0, 1, [255, 0, 0], thickness = 2, lineType = cv2.LINE_AA)

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
object_entering = {}
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
        filter_class = [0,1,3,4]
        memory = {}
        angle = -1
        already_counted = deque(maxlen=50)
        prev_frame_time=time.time()
        ratio =1
        #Area 1 
        p1_1=p1
        p2_1=p2
        p3_1=[p1[0]-70,abs(int((p1_1[1]-p4[1])/3))+p1[1]]
        p4_1=[p2[0]+70,abs(int((p2_1[1]-p3[1])/3))+p2[1]]
        Area_1=[p1_1,p2_1,p4_1,p3_1]
        
        #Area 3 
        p1_3=[p4[0],int(p1[1]/3)]
        p2_3=[p3[0],int(p2[1]/3)]
        p3_3=p3
        p4_3=p4
        Area_3=[p1_3,p2_3,p3_3,p4_3]
        
       
        
        outputs,bbox=self.detector.detect(img)
        output_new=[]
        cls=outputs[:,5]
        img_info["ratio"] = ratio
        filter_class = [0,1,3,4]
        
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
                origin_midpoint = (midpoint[0], img.shape[0] - midpoint[1]) # get midpoint respective to bottom-left
                re=cv2.pointPolygonTest(np.array(Area_1,np.int32),(int(origin_midpoint[0]),int(origin_midpoint[1])),False)
                if re>=0:
                    object_entering[str(tid)]=time.time()
              
                    
                
                

                
          
                cv2.circle(online_im,origin_midpoint,4,(255,25,24),3)

                if tid not in memory:
                    memory[tid] = deque(maxlen=2)

                memory[tid].append(midpoint)
                previous_midpoint = memory[tid][0]
            print(object_entering)       
                # origin_previous_midpoint = (previous_midpoint[0], img.shape[0] - previous_midpoint[1])
            if len(memory) > 50:
                del memory[list(memory)[0]]
            fps = 1/(time.time()-prev_frame_time) 
            # print(fps)
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1)
            #draw_Area_1
            cv2.polylines(online_im,[np.array(Area_1,np.int32)],isClosed=True,color=(0,255,0),thickness=2)
            # cv2.putText(online_im, "FPS: {:.2f}".format(fps),(50,100),cv2.FONT_HERSHEY_TRIPLEX,2,(255,0,0),1) 
            for cl, box in zip(cls,bbox) :
                cv2.putText(online_im,CLASS_NAME[int(cl)],(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,24,25),2)
                midpoint=(int((box[0]+box[2])/2),int((box[1]+box[3])/2))
                cv2.circle(online_im,midpoint,5,(255,25,24),cv2.FILLED)
                if int(cl)==2:
                    cs=True
            # print(center)
            
            for cl,id in zip(cls,online_ids):
                # print("id:",id)
                if str(id) not in M[CLASS_NAME[int(cl)]]:
                    M[CLASS_NAME[int(cl)]].append(str(id))
                if int(cl)==2 :
                    cs=True
            count_car=len(M["car"])
            count_bus=len(M["bus"])
            count_trailer=len(M["trailer"])
            count_truck=len(M["truck"])
            
            # print(online_midpoints_current)   
                
            # if cs :
            #     cv2.putText(online_im, "Person ",(450,100),cv2.FONT_HERSHEY_TRIPLEX,2,(0,0,255),1)
            # else :
            #     cv2.putText(online_im, " No Person ",(450,100),cv2.FONT_HERSHEY_TRIPLEX,2,(0,0,255),1)
        # print(M)
        return online_im,count_car,count_bus,count_trailer,count_truck,cs
        # else :
        #     return online_im
if __name__ == '__main__':

    cap=cv2.VideoCapture("video/video5.avi")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img = np.zeros((1280,720,3), np.uint8)
    track=Tracking()
    count=0
    check=0
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
           
            
            p1_new=[p1[0]+10,p1[1]+10]
            p2_new=[p2[0]+10,p2[1]+10]
            p3_new=[p3[0]+10,p3[1]+10]
            p4_new=[p4[0]+10,p4[1]+10]
            pts_new=np.array([p1_new,p2_new,p3_new,p4_new],np.int32)
            
            # ## (1) Crop the bounding rect
            # rect = cv2.boundingRect(pts)
            # x,y,w,h = rect
            # img_croped = img[y:y+h, x:x+w].copy()
            img_croped=get_area_detect(img,pts)
            ## (3) do bit-op
            # dst = cv2.bitwise_and(img_croped, img_croped, mask=mask)
            mask=find_lane_line(img_croped)
            center=[[abs(int((p1[0]+p2[0])/2)),abs(int((p1[1]+p2[1])/2))],[abs(int((p3[0]+p4[0])/2)),abs(int((p3[1]+p4[1])/2))]]
            # cv2.line(img_croped,center[0],center[1],(255,0,0),3)
            contours = find_contour(mask)
            for c in contours :
                if len(c)>20 and len(c)<100:
                    dist=calculate_distance(c[0][0],np.array(center))
                    if dist <30 :
                        c3_new.append(c)
                        cv2.drawContours(img_croped,c,-1,(0,0,255),3)
                        
            print(len(c3_new))

            # img_cropped=frame[p1[1]:p2[1],p1[0]:p2[0]]
            img_s,count_car,count_bus,count_trailer,count_truck,cs=track.infer(img_croped)
            cv2.putText(img,"Bus : {}".format(count_bus),(10,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            cv2.putText(img,"Car : {}".format(count_car),(10,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            cv2.putText(img,"Trailer : {}".format(count_trailer),(10,90),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            cv2.putText(img,"Truck : {}".format(count_truck),(10,110),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            if cs :
                cv2.putText(img,"Person : Found ",(10,130),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            else :
                cv2.putText(img,"Person : Not Found ",(10,130),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                
                
                # If a ROI is selected, draw it
                
            cv2.imshow("sss",img_s)
            # img[y:y+h, x:x+w]=img_s
        

        cv2.imshow('frame',img)
        # print(p1,p2)

        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        
        
        
        
