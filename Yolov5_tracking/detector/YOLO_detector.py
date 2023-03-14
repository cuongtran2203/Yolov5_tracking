import torch
import numpy as np
import onnxruntime
from .func import *

class Detector(object):
    def __init__(self, model='weights/yolov5n_12_2.onnx') -> None:
        self.providers = ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(model, providers=self.providers)
    
    def __pre_process(self, image):
        im = letterbox(image, 640, stride=32, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        return im

    def detect(self, image):
        blob = self.__pre_process(image)
        start=time.perf_counter()
        outputs = self.session.run(None, {self.session.get_inputs()[0].name:np.asarray(blob)})
        # print("time :{:.3f} s".format(time.perf_counter()-start))
        output_data = torch.tensor(outputs[0])
        y = non_max_suppression(output_data, 0.25, 0.45)[0]
        y[:, :4] = scale_boxes(blob.shape[2:], y[:, :4], image.shape).round()
        bbox=y[:, :4]
        print(bbox)
        # for box in bbox:
        #     cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),1)
        
        return y,bbox



