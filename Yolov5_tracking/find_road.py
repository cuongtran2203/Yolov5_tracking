import cv2 
import numpy as np
from lane_line_detector import *

# def nothing(x):
#     pass
# cv2.namedWindow("Trackbars")
# cv2.createTrackbar("L - H","Trackbars",0,179,nothing)
# cv2.createTrackbar("L - S","Trackbars",0,255,nothing)
# cv2.createTrackbar("L - V","Trackbars",0,255,nothing)
# cv2.createTrackbar("U - H","Trackbars",0,179,nothing)
# cv2.createTrackbar("U - S","Trackbars",0,255,nothing)
# cv2.createTrackbar("U - V","Trackbars",0,255,nothing)
# while True:
#     l_h=cv2.getTrackbarPos("L - H","Trackbars")
#     l_s=cv2.getTrackbarPos("L - S","Trackbars")
#     l_v=cv2.getTrackbarPos("L - V","Trackbars")
#     u_h=cv2.getTrackbarPos("U - H","Trackbars")
#     u_s=cv2.getTrackbarPos("U - S","Trackbars")
#     u_v=cv2.getTrackbarPos("U - V","Trackbars")
#     lower_color_road=np.array([l_h,l_s,l_v])
#     upper_color_road=np.array([u_h,u_s,u_v])
   
    
#     img=cv2.imread("test3.jpg")
#     hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     mask=cv2.inRange(hsv,lower_color_road,upper_color_road)
#     cv2.imshow("frame",mask)
#     cv2.imshow("mask",hsv)
#     key=cv2.waitKey(1)
#     if key ==ord("q"):
#         break
###################################################################
p1,p2,p3,p4 =None,None,None,None 
state=0
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

if __name__=='__main__':

    vid=cv2.VideoCapture("video/video13.avi")
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)
    while True:
        _,frame=vid.read()
        frame=cv2.resize(frame,(1280,720))
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_2 = cv2.GaussianBlur(frame,(5,5),0)
        if p1 is not None and p2 is not None and p3 is not None and p4 is not None :
            pts=np.array([p1,p2,p3,p4],np.int32)
            img_croped=get_area_detect(frame_2,pts)
            img_copy=img_croped.copy()
            mask=road_detection(img_croped)
            mask2=mask.copy()
            kernel = np.ones((5,5),np.uint8)

            # Remove unnecessary noise from mask

            # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            arr_point=[p1,p2,p3,p4]
            #sort array from y
            arr_point.sort(key=lambda x:x[1])
            start_point=arr_point[0]
            pts2_array=np.array(arr_point)
            rect = cv2.boundingRect(pts2_array)
            x,y,w,h = rect
            arr1= mask2[y:y+h, x:x+w].copy()
            arr1=np.where(arr1==0,255,0)
            end_point=arr_point[3]
            print(mask2.shape)
            # # for idx_x,x in enumerate(mask2.shape[0]):
            # #     for idx_y,y in enumerate(mask2.shape[1]):
            # #         if idx_x <
            # arr1=mask2[start_point[1]:end_point[1],start_point[0]:end_point[0]]
            # arr1=np.where(arr1==0,255,0)
            mask2[y:y+h, x:x+w]=arr1
            # mask2=np.where(mask2==0,255,0)
            mask2=mask2.astype(np.uint8)
            color_object= np.array([255,255,24], dtype='uint8')
            # road_img=frame[start_point[1]:end_point[1],start_point[0]:end_point[0]]
            masked_img2 = np.where(mask2[...,None], color_object, frame)
            pts = pts.reshape((-1, 1, 2))
 
            # isClosed = True
            
            # Blue color in BGR
            color = (0, 0,255)
            
            # Line thickness of 2 px
            thickness = 2
            isClosed = True
            # Using cv2.polylines() method
            # Draw a Blue polygon with
            # thickness of 1 px
            masked_img2= cv2.polylines(masked_img2, [pts],
                                isClosed, color, thickness)
            
            
            # use `addWeighted` to blend the two images
            # the object will be tinted toward `color`
            # out2 = cv2.addWeighted(frame, 0.8, masked_img2, 0.2,0)
            cv2.imshow("skks",masked_img2)
            # color to fill
            # color_road = np.array([0,255,0], dtype='uint8')
            # # equal color where mask, else image
            # # this would paint your object silhouette entirely with `color`
            # masked_img = np.where(mask[...,None], color_road, frame)
            # # use `addWeighted` to blend the two images
            # # the object will be tinted toward `color`
            # out = cv2.addWeighted(frame, 0.8, masked_img, 0.2,0)
            # cv2.imshow("ssss",masked_img)
        cv2.imshow("frame",frame)
        key=cv2.waitKey(100)
        if key==ord("q"):
            break
    
