
import numpy as np
import cv2
import time
from src.counting import *
from src.visualize import plot_tracking
from tracker.byte_tracker import BYTETracker
from detector.YOLO_detector import Detector
from detector.func import *
from lane_line_detector import *
import time
import math
import matplotlib.pyplot as plt
from config.config_cam import *
p1, p2,p3,p4 = None, None,None,None
state = 0
frame =None
crop = False
CLASS_NAME=["bus","car","lane","person","trailer","truck","bike"]
def get_area_detect(img, points):
    # points = points.reshape((-1, 1, 2))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dts = cv2.bitwise_and(img, img, mask=mask)
    return dts
# Called every time a mouse event happen
def on_mouse(event, x, y, flags, userdata):
    global state, p1, p2,p3,p4
    # Left click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Select first point
        if state == 0:
            p1 = [x,y]
            state += 1
        # Select second point
        elif state == 1:
            p2 = [x,y]
            state += 1
          # Select second point
        elif state == 2:
            p3 = [x,y]
            state += 1
          # Select second point
        elif state == 3:
            p4 = [x,y]
            state += 1
    # Right click (erase current ROI)
    if event == cv2.EVENT_LBUTTONDBLCLK:
        p1, p2,p3,p4 = None, None
        state = 0

class Args():
    def __init__(self) -> None:
        self.track_thresh = 0.4
        self.track_buffer = 30
        self.match_thresh = 0.7
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
        cs=False
        # create filter class
        filter_class = [0]
        ratio =1  
        outputs,bbox=self.detector.detect(img)
        # print("img shape :",img.shape)
        cls=outputs[:,5]
        img_info["ratio"] = ratio
        filter_class = [0,1,3,4,5,6]
        list_count=[0,0,0,0]
        if outputs is not None:
            online_targets = self.tracker.update(outputs, [height,width], self.test_size, filter_class)
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
            for cl, id in zip(cls,online_ids) :
                if str(id) not in M[CLASS_NAME[int(cl)]]:
                    M[CLASS_NAME[int(cl)]].append(str(id))
                if int(cl)==2 :
                    cs=True
            list_count=[len(M["car"]),len(M["bus"]),len(M["trailer"]),len(M["truck"])]
        return list_count,cs,bbox,cls,online_ids

if __name__ == '__main__':
    cap=cv2.VideoCapture("./video/video10.avi")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img = np.zeros((1280,720,3), np.uint8)
    track=Tracking()
    count=0
    check=0
    online_ids =[]
    start_point={}
    
    check_point={}
    update_point_t1={}
    M={"bus":[],"car":[],"trailer":[],"truck":[],"person":[],"lane":[],"bike":[]}
    Max_contours=0
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)
    while True:
        update_point={}
        _,frame=cap.read()
        c3_new=[]
        count+=1
        if count>length:
            break
        frame=cv2.resize(frame,(1280,720))
        img=frame.copy()
        start_time=time.time()
        # Cropping image
        if p1 is not None and p2 is not None and p3 is not None and p4 is not None :
            pts=np.array([p1,p2,p3,p4],np.int32)
            #crop frame 
            img_croped=get_area_detect(img,pts)
            cv2.polylines(img,[pts],True,(0,0,142),3)
            img_copy=img_croped.copy()
            #Tracking
            list_count,cs,bbox,cls,online_ids=track.infer(img_croped)
            for box in bbox :
                cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,125),2)
                box=list(map(int,box))
                img_vehicle=np.zeros([box[3]-box[1],box[2]-box[0],3],dtype=np.uint8,order='C')
                img_copy[box[1]:box[3],box[0]:box[2]]=img_vehicle
            mask=find_lane_line(img_copy)
            arr_point=[p1,p2,p3,p4]
            arr_point_1=[p1,p2,p3,p4]
            arr_point_1.sort()
            #sort array from y
            arr_point.sort(key=lambda x:x[1])
            point_1=[int((arr_point[0][0]+arr_point[1][0])/2),int((arr_point[0][1]+arr_point[1][1])/2)]
            point_2=[int((arr_point[2][0]+arr_point[3][0])/2),int((arr_point[2][1]+arr_point[3][1])/2)]
            center=[point_1,point_2]
            contours = find_contour(mask)
            for c in contours :
                if len(c)>20 and len(c)<100:
                    dist=calculate_distance(c[0][0],np.array(center))
                    x,y,w,h=cv2.boundingRect(c)
                    img_contour=np.zeros([h,w,3],dtype=np.uint8,order="C")
                    img_copy[y:y+h,x:x+w]=img_contour
                    if dist <60 :
                        c3_new.append(c)
                        cv2.drawContours(img,c,-1,(0,0,255),3)
            cv2.putText(img,"Bus : {}".format(list_count[1]),(10,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            cv2.putText(img,"Car : {}".format(list_count[0]),(10,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            cv2.putText(img,"Trailer : {}".format(list_count[2]),(10,90),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            cv2.putText(img,"Truck : {}".format(list_count[3]),(10,110),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            cv2.putText(img,"SUM OUT : {}".format(sum(list_count)),(10,150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,125,255),1)
            if cs :
                cv2.putText(img,"Person : Found ",(10,130),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            else :
                cv2.putText(img,"Person : Not Found ",(10,130),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            print(" Have {} object in region".format(len(bbox)))
            for box,cl,id in zip(bbox,cls,online_ids):
                if int(cl)!=2 :
                    mid_point=(int((box[0]+box[2])/2),int((box[1]+box[3])/2))
                    if str(id) not in start_point.keys():
                        start_point[str(id)]=[mid_point,time.time()]
                        print("Add object to dict")
                    update_point[str(id)]=[mid_point,time.time()]

                
            list_start_point_keys=list(start_point.keys())
            list_update_point_keys=list(update_point.keys())
            
            
            
            '''
            Hàm xử lý khi đã có 3 mảng
            '''
            '''
            Khởi tạo 1 biến bool để làm chậm 1 nhịp cập nhật value trong dict 
            
            '''
            if check %2 ==0 :
                update_point_t1=update_point.copy()
                check +=1 
            '''
            Phát hiện đối tượng đi ra khỏi vùng quan sát 
            '''
            print(" Có  {} đối tượng đã đi ra khỏi vùng quan sát ".format(len(list_start_point_keys)-len(list_update_point_keys)))
            print("Current ",list_update_point_keys)
            print("Previous ",list_start_point_keys)
            print("At time t -1 :",list(update_point_t1.keys()))
            # if list_start_point_keys!= list_update_point_keys :
            #     print("Have some objects disappear")
                  
        cv2.imshow('frame',img)
        ##### Clear dict update_point_t1 sau khi đã sử dụng ###########
        
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        
        
        
