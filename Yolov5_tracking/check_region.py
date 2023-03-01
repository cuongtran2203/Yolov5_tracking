import cv2
import numpy as np
from detector.YOLO_detector import Detector
# Called every time a mouse event happen
def get_area_detect(img, points):
    # points = points.reshape((-1, 1, 2))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dts = cv2.bitwise_and(img, img, mask=mask)
    return dts
p1, p2,p3,p4 = None, None,None,None
state = 0
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
if __name__ == "__main__":
    cap=cv2.VideoCapture("./video/video5.avi")
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)
    model=Detector()
    while True :
        _,frame=cap.read()
        frame=cv2.resize(frame,(1280,720))
        vehicles_enter_A={}
        if p1 is not None and p2 is not None and p3 is not None and p4 is not None :
            arr_point=[p1,p2,p3,p4]
            pts=np.array([p1,p2,p3,p4],np.int32)
            cv2.polylines(frame,[pts],True,(0,0,142),3)
            img_croped=get_area_detect(frame,pts)
            img_copy=img_croped.copy()
            output,bbox=model.detect(img_croped)
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
            Area_C.sort(key=lambda x:x[1])
            # print("C Area",Area_C)
            np_Area_A=np.array([Area_A[0],Area_A[1],Area_A[3],Area_A[2]],np.int32)
            np_Area_B=np.array([Area_B[1],Area_B[0],Area_B[2],Area_B[3]],np.int32)
            np_Area_C=np.array([Area_C[0],Area_C[1],Area_C[3],Area_C[2]],np.int32)
            for box in bbox:
                mid_point=(int((box[0]+box[2])/2),int((box[1]+box[3])/2))
                print(mid_point)
                
                result_A=cv2.pointPolygonTest(np_Area_A,mid_point,False)
                if result_A >=0:
                    print("Object appeared at Area A")
                result_B=cv2.pointPolygonTest(np_Area_B,mid_point,False)
                if result_B >=0:
                    print("Object appeared at Area B")
                result_C=cv2.pointPolygonTest(np_Area_C,mid_point,False)
                if result_C >=0:
                    print("Object appeared at Area C")
                # print(result)

            cv2.polylines(frame,[np_Area_A],True,(125,0,142),3)
            cv2.polylines(frame,[np_Area_B],True,(0,100,142),3)
            cv2.polylines(frame,[np_Area_C],True,(250,120,142),3)
            
        cv2.imshow("frame",frame)
        key=cv2.waitKey(25)
        if key==ord('q'):
            break
        
        
            
            