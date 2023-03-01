import cv2
import numpy as np
from lane_line_detector import *
from detector.YOLO_detector import Detector
p1,p2,p3,p4 =None,None,None,None 
state=0
def get_area_detect(img, points):
    # points = points.reshape((-1, 1, 2))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dts = cv2.bitwise_and(img, img, mask=mask)
    return dts

def on_mouse(event, x, y, flags, userdata):
    global state, p1, p2,p3,p4
    # Left click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Select first point
        if state == 0:
            p1 = [x,y]
            state += 1
            print(p1)
        # Select second point
        elif state == 1:
            p2 = [x,y]
            state += 1
            print(p2)
          # Select second point
        elif state == 2:
            p3 = [x,y]
            state += 1
            print(p3)
          # Select second point
        elif state == 3:
            p4 = [x,y]
            state += 1
            print(p4)
    # Right click (erase current ROI)
    if event == cv2.EVENT_LBUTTONDBLCLK:
        p1, p2,p3,p4 = None, None
        state = 0
if __name__=="__main__":
    # img=cv2.imread("test6.jpg")
    # gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur=cv2.GaussianBlur(gray,(5,5),0)
    model=Detector()
    cap=cv2.VideoCapture("video/video13.avi")
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)
    while True :
        _,img=cap.read()
        img=cv2.resize(img,(1280,720))
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gray,(7,7),0)
        if p1 is not None and p2 is not None and p3 is not None and p4 is not None :
            pts=np.array([p1,p2,p3,p4],np.int32)
            img_croped=get_area_detect(blur,pts)
            img_croped_det=get_area_detect(img,pts)
            img_copy=img_croped.copy()
            output,bbox=model.detect(img_croped_det)
            for box in bbox :
                cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,125),2)
                box=list(map(int,box))
                img_vehicle=np.zeros([box[3]-box[1],box[2]-box[0]],dtype=np.uint8,order='C')
                img_croped[box[1]:box[3],box[0]:box[2]]=img_vehicle
                
            
            kernel = np.ones((5,5),np.uint8)
            cv2.polylines(img,[pts],True,(0,0,145),3)
            # Remove unnecessary noise from mask
            # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            arr_point=[p1,p2,p3,p4]
            #sort array from y
            arr_point.sort(key=lambda x:x[1])
            point_1=[int((arr_point[0][0]+arr_point[1][0])/2),int((arr_point[0][1]+arr_point[1][1])/2)]
            point_2=[int((arr_point[2][0]+arr_point[3][0])/2),int((arr_point[2][1]+arr_point[3][1])/2)]
            center=[point_1,point_2]
            start_point=arr_point[0]
            canny=cv2.Canny(img_croped,100,300)
            # thresh = cv2.adaptiveThreshold(img_croped, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
            
            contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                # cnt = contours[4]
                if len(c)>20 and len(c)<100:
                    dist=calculate_distance(c[0][0],np.array(center))
                    x,y,w,h=cv2.boundingRect(c)
                    if dist>60 and dist<150:
                        cv2.drawContours(img, c, -1, (0,255,0), 3)
                        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,25,10),2)
            # cv2.imshow("ssss",canny)
        cv2.imshow("frame", img)
        key=cv2.waitKey(100)
        if key==ord("q"):
            break