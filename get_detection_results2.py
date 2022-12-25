import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import os
import numpy as np
import matplotlib.pyplot as plt


random.seed(1)

classes_str = ['Phone']

image_directory = "mAP-master/input/images-optional"

# Initialize
set_logging()
device = select_device('')
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load('yolov7x_phone.pt', map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size

model = TracedModel(model, device, imgsz)

if half:
    model.half()  # to FP16

def detect(model, image_path, image_name, imgsz=640, augment=True, conf_thresh=0.3, iou_thresh=0.45):
    # source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    # save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    

    # Set Dataloader
    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # else:
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    # start_time = 0
    for path, img, im0s, vid_cap in dataset:
        start_time = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=True)[0]
                
        
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
            # print(f"Results first: {pred}")
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=0, agnostic=True)
        t3 = time_synchronized()
        
        print(pred)
        
        # print(pred)
        # if len(pred[0])> 0:
            
        #     output = pred[0].tolist()[0]
        #     output = [classes[int(output[-1])]] + [output[-2]] + output[:4]
        #     print(f"Results: {output}")
        
        # else:
        #     output 
        
        #Save detection results in the required order/format
        myfile = open(f'mAP-master/input/detection-results/{image_name}.txt', 'w')
        
        
        for det in pred:
            if len(det)>0:
                print(det)       
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                print(det)
                output = det[:, :].tolist()[0]
                print(output)
                output = [classes_str[int(output[-1])]] + [round(output[-2],3)] + output[:4]
                output = [str(x) for x in output]
                output = ' '.join(output)
                output +='\n'
                print(f"Results: {output}")
                myfile.write(output)
                
        myfile.close()           
        
        # for detection in pred[0]:
        #     if len(detection)>0:
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        #         output = detection.tolist()
                # output = [classes_str[int(output[-1])]] + [round(output[-2],3)] + output[:4]
                # output = [str(x) for x in output]
                # output = ' '.join(output)
                # output +='\n'
                # print(f"Results: {output}")
                # myfile.write(output)              
        #     else:
        #         print(pred)
            
        # myfile.close()   


for filename in os.listdir(image_directory):
    if filename.endswith(".JPG") or filename.endswith(".JPEG") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        #detection = detection_results(yolo, filename)
        image_name = Path(os.path.join(image_directory, filename)).stem
        detection = detect(model, os.path.join(image_directory, filename), image_name)
print('All detection results now saved!')

