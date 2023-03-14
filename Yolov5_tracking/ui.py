from tkinter import *
from tkinter import ttk
import cv2
import numpy as np
from PIL import ImageTk
from math import sqrt
import sys
from detector.YOLO_detector import Detector
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor 

class FrameProcess:
    @staticmethod
    def read_video(path):
        videos = []
        for p in path:
            cap = cv2.VideoCapture(p)
            videos.append(cap)
        return videos 
    
    @staticmethod
    def nearest_square_number(n):
        return int(sqrt(n))+1

    @staticmethod
    def square_number(n):
        sqr = sqrt(n)
        if (sqr * sqr) == n:
            return True
        else:
            return False
        
    @staticmethod
    def process_window(image_list):
        image_length = len(image_list)
        if FrameProcess.square_number(image_length):
            stream_tile_number = int(sqrt(image_length))
            new_width = int(1920 / stream_tile_number)
            new_height = int(1080 / stream_tile_number)
            resized_image = [cv2.resize(image, (new_width, new_height)) for image in image_list]
            output_image = cv2.vconcat([cv2.hconcat(resized_image[i:i+stream_tile_number]) for i in range(0, image_length, stream_tile_number)])
        else:
            stream_tile_number = FrameProcess.nearest_square_number(image_length)
            expected_image_length = stream_tile_number**2
            blank_tile_number = expected_image_length - image_length
            new_width = int(1920 / stream_tile_number)
            new_height = int(1080 / stream_tile_number)
            blank_image = np.zeros((new_height, new_width, 3), np.uint8)
            image_list.extend([blank_image] * blank_tile_number)
            resized_image = [cv2.resize(image, (new_width, new_height)) for image in image_list]
            output_image = cv2.vconcat([cv2.hconcat(resized_image[i:i+stream_tile_number]) for i in range(0, expected_image_length, stream_tile_number)])
        return output_image

class App(Frame):
    
    def __init__(self, parent):
        Frame.__init__(self, parent, background="white")
        self.parent = parent
        self.initUI()
        self.model=Detector()
        print("Loading model...")
        
    def initUI(self):
        self.parent.title("Window")
        self.pack(fill=BOTH, expand=1)
        AddButton = Button(self, text="Add Link", command=self.add)
        AddButton.place(x=50, y=250)
        Summit_button=Button(self,text="Summit",command=self.submit_link)
        Summit_button.place(x=150,y=250)
        
        self.image_label = Label(self)
    def process_excute(self,frame):
        output=self.model.detect(frame)
        # tasks_that_are_done.put(output)
        return output
        
    def add(self):
        self.Label=Label(self,text="Enter link source or RTSP link")
        self.Label.place(x=50,y=20)
        self.link_entry = Entry(self)
        self.link_entry.place(x=50, y=50)
        
          
    
    def submit_link(self):
        links = self.link_entry.get().split(",")
        print("Submitted links: ", links)
        video_frames = FrameProcess.read_video(links)
        while True:
            frames = []
            for cap in video_frames:
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    p=Process(target=self.model.detect,args=(frame,))
                    p.start()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                    frames.append(frame)
            # with ProcessPoolExecutor(max_workers=7) as executor:
            #     result= executor.map(self.process_excute,frames)
            show_image = FrameProcess.process_window(frames)
            cv2.imshow("image", show_image)
            if cv2.waitKey(25) == ord('q'):
                break
        for cap in video_frames:
            cap.release()
        cv2.destroyAllWindows()
    
        self.link_entry.delete(0, END)

        
root = Tk()
root.geometry("1280x720")
app = App(root)
root.mainloop()