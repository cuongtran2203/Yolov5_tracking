import cv2
import argparse
import numpy as np

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(cv2.__version__)
def tlbr_to_tlwh(tlbr):
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    return ret
def make_parser():
    parser=argparse.ArgumentParser("Tracking Opencv")
    parser.add_argument("--tracker_type",type=str,default="KCF",help="select tracking algorithm")
def tracking(args,img,output):
    #output has format [x1,y1,x2,y2,score,class]
    output=output.cpu().detach().numpy()
    scores = output[:, 4] *output[:,5]
    box = output[:, :4]  # x1y1x2y2
    cls = output[:, 5]
    

    
    
    if args.tracker_type=="KCF":
        tracker = cv2.TrackerKCF_create()
    if args.tracker_type=="BOOSTING":
        tracker = cv2.TrackerBoosting_create()
    if args.tracker_type=="TLD":
        tracker =cv2.TrackerTLD_create()
    if args.tracker_type=="MIL":
        tracker = cv2.TrackerMIL_create()
    if args.tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if args.tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    if args.tracker_type == 'MOSSE':
        tracker = cv2.legacy_TrackerMOSSE.create()
    ok=tracker.init(img,box)
     # Start timer
    # timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(img)

    # Calculate Frames per second (FPS)
    # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(img, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    cv2.putText(img, args.tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    return img
    
    
    
    

    

    

    

    