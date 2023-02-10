import cv2
import numpy as np
from lane_line_detector import *
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
    cap=cv2.VideoCapture("video/video14.avi")
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)
    while True :
        _,img=cap.read()
        img=cv2.resize(img,(1280,720))
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gray,(5,5),0)
        if p1 is not None and p2 is not None and p3 is not None and p4 is not None :
            pts=np.array([p1,p2,p3,p4],np.int32)
            img_croped=get_area_detect(blur,pts)
            img_copy=img_croped.copy()
            kernel = np.ones((5,5),np.uint8)
            # Remove unnecessary noise from mask
            # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            arr_point=[p1,p2,p3,p4]
            #sort array from y
            arr_point.sort(key=lambda x:x[1])
            start_point=arr_point[0]
            thresh = cv2.adaptiveThreshold(img_croped, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
            contours=find_contour(thresh)
            
            print(thresh)
            ### draw contours 
            for c in contours :
                    # img_contour=np.zeros([h,w,3],dtype=np.uint8,order="C")
                    # img_copy[y:y+h,x:x+w]=img_contour
                cv2.drawContours(img,c,-1,(0,0,255),3)
                # cv2.circle(img,c[0][0],10,(255,0.120),-1)
            # cv2.imshow("ssssa",img_copy)
            cv2.imshow("ssss",thresh)
        cv2.imshow("frame", img)
        key=cv2.waitKey(100)
        if key==ord("q"):
            break