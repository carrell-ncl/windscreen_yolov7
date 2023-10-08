"""
@author Steven Carrell
@email - steven.carrell.ncl@gmail.com
Cloned and modified from - https://github.com/WongKinYiu/yolov7
"""

import time
from pathlib import Path

# import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    TracedModel,
)
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt


random.seed(1)

classes_str = ["Phone"]

# image_directory = "mAP-master/input/images-optional"
image_directory = (
    r"C:\Users\Steve\Desktop\phone_weather_conditions\direct_sun/input/images-optional"
)
# out_folder = 'mAP-master/input/detection-results/'
out_folder = r"C:\Users\Steve\Desktop\phone_weather_conditions/direct_sun/input/detection-results/"

# Checks for directories and creates if false
if not os.path.isdir(out_folder):
    os.makedirs(out_folder)

# Initialize
set_logging()
device = select_device("")
half = device.type != "cpu"  # half precision only supported on CUDA

# Load model
model = attempt_load("yolov7x_phone_best.pt", map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size

model = TracedModel(model, device, imgsz)

if half:
    model.half()  # to FP16


def detect(
    model: TracedModel,
    image_path: str,
    image_name: str,
    imgsz: int = 640,
    augment: bool = True,
    conf_thresh: float = 0.1,
    iou_thresh: float = 0.45,
):
    """_summary_

    Args:
        model (TracedModel): Custom Torch model
        image_path (str): Location of image
        image_name (str): Image filename exclusing extension
        imgsz (int, optional): Image size. Defaults to 640.
        augment (bool, optional): _description_. Defaults to True.
        conf_thresh (float, optional): Confidence threshold value. Defaults to 0.1.
        iou_thresh (float, optional): IOU threshold value. Defaults to 0.45.
    """

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
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
        if device.type != "cpu" and (
            old_img_b != img.shape[0]
            or old_img_h != img.shape[2]
            or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=True)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thresh, iou_thresh, classes=0, agnostic=True
        )
        t3 = time_synchronized()

        myfile = open(f"{out_folder}{image_name}.txt", "w")

        for det in pred:
            if len(det) > 0:
                print(f"DET1: {det}")
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                print(f"DET2: {det}")
                output = det[:, :].tolist()[0]
                output = (
                    [classes_str[int(output[-1])]] + [round(output[-2], 3)] + output[:4]
                )
                output = [str(x) for x in output]
                output = " ".join(output)
                output += "\n"
                print(f"Results: {output}")
                myfile.write(output)

        myfile.close()


for filename in os.listdir(image_directory):
    if (
        filename.endswith(".JPG")
        or filename.endswith(".JPEG")
        or filename.endswith(".jpg")
        or filename.endswith(".jpeg")
    ):
        # detection = detection_results(yolo, filename)
        image_name = Path(os.path.join(image_directory, filename)).stem
        detection = detect(model, os.path.join(image_directory, filename), image_name)
print("All detection results now saved!")
