
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
# def estimate_velocity(A=[None,None],B=[None,None],delta_t=None) :
#     # distance_max_pixel=max(math.hypot(abs(p1[0]-p4[0]),abs(p1[1]-p4[1])),math.hypot(abs(p2[0]-p3[0]),abs(p2[1]-p3[1])))
#     # distance_constant=distance_max_pixel*ratio
#     # meters_per_pixel=distance_constant/distance_max_pixel
#     distance_in_pixel=math.hypot(abs(A[0]-B[0]),abs(A[1]-B[1]))
  
#     distance_in_meter_zone=ratio*distance_in_pixel
#     speed=distance_in_meter_zone/delta_t

#     return speed*math.cos(math.pi/3)
    
 
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
        frame_id = 0
        cs=False
        # create filter class
        filter_class = [0]
        ratio =1  
        start_time=time.time()
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
            for t,box in zip(online_targets,bbox):
                tlwh = t.tlwh
                tid = t.track_id              
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                midpoint = tlbr_midpoint(tlwh)
                # print("midpoint",mid_p)
                origin_midpoint = (midpoint[0], img.shape[0] - midpoint[1]) # get midpoint respective to bottom-left
                info_list_id=[origin_midpoint,time.time()]#save info ids [ (cx,cy),time]
                current_dict_info.update({str(tid):info_list_id})  
            online_im= plot_tracking(img, online_tlwhs, online_ids, frame_id=frame_id + 1)
            for cl, id,box,speed in zip(cls,online_ids,bbox,speed_list) :
                if str(id) not in M[CLASS_NAME[int(cl)]]:
                    M[CLASS_NAME[int(cl)]].append(str(id))
                if int(cl)==2 :
                    cs=True
                else :
                    if int(cl)!=2 and speed>1:
                        cv2.putText(frame,"{:.2f} km/h".format(speed),(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,24,25),2)
            speed_list.clear()
            list_count=[len(M["car"]),len(M["bus"]),len(M["trailer"]),len(M["truck"])]
        fps=1/(time.time()-start_time) 
        # print(fps)
        speed_list.clear()
        return frame,list_count,cs,current_dict_info,bbox,fps,cls,online_ids

if __name__ == '__main__':
    prev_dict_id ={}
    cap=cv2.VideoCapture("./video/video5.avi")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img = np.zeros((1280,720,3), np.uint8)
    track=Tracking()
    count=0
    check=0
    speed_list=[]
    np_Area_A,np_Area_B,np_Area_C=None,None,None
    enter_areaA={}
    enter_areaB={}
    mid_point_ereA={}
    mid_point_ereB={}
    M={"bus":[],"car":[],"trailer":[],"truck":[],"person":[],"lane":[],"bike":[]}
    Max_contours=0
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
        start_time=time.time()
        # Cropping image
        if p1 is not None and p2 is not None and p3 is not None and p4 is not None :
            pts=np.array([p1,p2,p3,p4],np.int32)
            #crop frame 
            img_croped=get_area_detect(img,pts)
            cv2.polylines(img,[pts],True,(0,0,142),3)
            img_copy=img_croped.copy()
            #Tracking
            img_s,list_count,cs,dict_info,bbox,fps,cls,online_ids=track.infer(img_croped)
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
            A=[arr_point[3][0],arr_point[3][1]*2/3]
            B=[arr_point[2][0],arr_point[2][1]*2/3]
            C=[arr_point[1][0]+100,arr_point[1][1]*1.6]
            D=[arr_point[0][0]-100,arr_point[0][1]*1.6]
            Area_A=[arr_point[3],arr_point[2],A,B]
            Area_A.sort(key=lambda x:x[1])
            # print('A area',Area_A)
            Area_B=[C,D,arr_point[1],arr_point[0]]
            Area_B.sort(key=lambda x:x[1])
            # print('B Area',Area_B)
            Area_C=[A,B,C,D]
            Area_C.sort()
            # print("C Area",Area_C)
            np_Area_A=np.array([Area_A[0],Area_A[1],Area_A[3],Area_A[2]],np.int32)
            np_Area_B=np.array([Area_B[1],Area_B[0],Area_B[2],Area_B[3]],np.int32)
            np_Area_C=np.array([Area_C[0],Area_C[1],Area_C[3],Area_C[2]],np.int32)
            # cv2.polylines(img,[np_Area_A],True,(125,0,142),3)
            # cv2.polylines(img,[np_Area_B],True,(0,100,142),3)
            # cv2.polylines(img,[np_Area_C],True,(250,120,142),3)
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
            for box,cl,id in zip(bbox,cls,online_ids):
                if int(cl)!=2 :
                    
                    mid_point=(int((box[0]+box[2])/2),int((box[1]+box[3])/2))
                    result_B=cv2.pointPolygonTest(np_Area_B,mid_point,False)
                    if result_B >=0:
                        enter_areaB[str(id)]=time.time()
                        mid_point_ereB[str(id)]=mid_point
                    result_A=cv2.pointPolygonTest(np_Area_A,mid_point,False)
                    if result_A >=0:
                        if str(id) in enter_areaB.keys():
                            enter_areaA[str(id)]=time.time()-enter_areaB[str(id)]
                            mid_point_ereA[str(id)]=mid_point
                            point_B= mid_point_ereB[str(id)]
                            distance_in_pixel=math.hypot(abs(mid_point[0]-point_B[0]),abs(mid_point[1]-point_B[1]))
                            '''
                            Ở đây ta tính tỷ lệ pixels trên khung hình là 4 pixels /mm 
                            Thời gian được tính là ms
                            Công thức tính khoảng cách thực = tỷ lệ pixels* khoảng cách giữa 2 điểm trên khung hình
                            '''
                            constant_distance=distance_in_pixel/4 #met
                            velocity=(constant_distance/enter_areaA[str(id)])*3.6
                            print("velocity : ",velocity)
                            cv2.putText(img,"{:.2f} km/h".format(velocity),(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,24,25),2)
                    '''
                    Để phát hiện xem xe có dừng đỗ hay không t có thể dựa vào thời gian tồn tại trong vùng không gian đó 
                    
                    '''
         
                    result_C=cv2.pointPolygonTest(np_Area_C,mid_point,False)
            if len(dict_info)>0:
                prev_dict_id=dict_info.copy()
        cv2.imshow('frame',img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        
        
        
