import argparse
import time
from pathlib import Path

import queue
import cv2
import numpy
from numpy import random

import sys
sys.path.insert(0, '/home/gionji/yolov7')

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class YOLOv7Detector:
    def __init__(self, weights, classes = None, img_size=640, device='0', save_img=False, 
                 view_img = False, save_txt = False, conf_thres = 0.25, iou_thres = 0.45,  
                 agnostic_nms = False, augment = False, update = False, project = 'runs/detect',
                 name = 'exp', exist_ok = False, no_trace = True, text_path = './yolo_text_path_idk'
                 ):
        
        self.weights = weights
        self.img_size = img_size
        self.device = device
        self.save_img = save_img
        self.view_img = view_img
        self.save_txt = save_txt
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.no_trace = no_trace
        self.text_path = text_path

        self.initialize_detector()

    ## un-used
    def initialize_detector(self):
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'
        #print('device and half: ',self.device, self.half)
        
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)
        self.stride = int( self.model.stride.max() )
        self.img_size = check_img_size(self.img_size, s=self.stride)

        if self.half:
            self.model.half()  # to FP16


    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = numpy.mod(dw, stride), numpy.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


    def detect(self, image, save_img=False):
        ### opt are the parsed parameters.
        weights, view_img, save_txt, imgsz, trace =  self.weights,  self.view_img,  self.save_txt,  self.img_size, not self.no_trace
        
        # Directories
        #save_dir = Path(increment_path(Path( self.project) /  self.name, exist_ok= self.exist_ok))  # increment run
        #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        #set_logging()

        if trace:
            self.model = TracedModel(self.model, self.device,  self.img_size)

        if self.half:
            self.model.half()  # to FP16

        ## Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        
        # Reshape the raw data into an RGB array
        #im0 = numpy.reshape(numpy.copy(image.raw_data), (image.height, image.width, 4)) 
        
        image = cv2.resize(image, (self.img_size, self.img_size))
        image_h = image.shape[0]
        image_w = image.shape[1]

        im0 = numpy.reshape( numpy.copy(image), (image_h, image_w, image.shape[2]) ) 
        im0 = im0[:, :, :3]
        
        # Padded resize
        img = self.letterbox(im0)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = numpy.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=  self.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment= self.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, 
                                   self.conf_thres, 
                                   self.iou_thres, 
                                   classes=self.classes, 
                                   agnostic=self.agnostic_nms)
        t3 = time_synchronized()

        im0s = None # try to fix it
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            #s, im0, frame = '', im0, image.frame
            s, im0 = '', im0

            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if  self.save_conf else (cls, *xywh)  # label format
                        with open( self.txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                
                    label = f'{names[int(cls)]} {conf:.2f}'
                    im0 = numpy.ascontiguousarray(im0, dtype=numpy.uint8)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        #if self.save_txt or self.save_img:
            #s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #print(f"Results saved to {save_dir}{s}")

        #print(f'Done. ({time.time() - t0:.3f}s)')

        return im0, pred

        '''
        with torch.no_grad():
            if  self.update:  # update all models (to fix SourceChangeWarning)
                for  self.weights in ['yolov7.pt']:
                    detect()
                    strip_optimizer(opt.weights)
            else:
                detect()
        '''