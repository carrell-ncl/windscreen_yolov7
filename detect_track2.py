import os
import cv2
import time
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
                time_synchronized, TracedModel
# from utils.download_weights import download

#For SORT tracking
import skimage
from sort import *

#............................... Tracker Functions ............................
""" Random created palette"""
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

"""" Calculates the relative bounding box from absolute pixel values. """
def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, names=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id) + ":"+ names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)
    return img

def draw_boxes_second(img, bbox,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)

    return img
#..............................................................................


def detect(save_img=False):
    
    source, weights, weights_second, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim= opt.source, opt.weights, opt.weights_second, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.colored_trk, opt.save_bbox_dim
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))


    #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    #......................... 
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    
    # Load second stage model if required
    if len(weights_second)>0:
            # Load model
            print('Loading second model')
            model_second = attempt_load(weights_second, map_location=device)  # load FP32 model
        

    if trace:
        model = TracedModel(model, device, opt.img_size)
        
    # print(f"model1: {model}")
    # print(f"model2: {model_second}")

    if half:
        model.half()  # to FP16
        model_second.half()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    print(f"Names: {names}")

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        model_second(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_second.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    #........Rand Color for every trk.......
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #.........................
   
    # Stores fps to smooth out value
    fps_list_track = [] 
   
    for path, img, im0s, vid_cap in dataset:
        start_time = time.time()
        # print(type(im0s))
        # print(im0s)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # print(img)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]
                # model_second(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # pred2 = model_second(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
            
        # for i, detection in enumerate(pred[0]):
        #     if len(detection)>0:
        #         output = detection.tolist()[:4]
        #         output = [int(x) for x in output]
        #         cropped = im0s[output[1]:output[3], output[0]:output[2]]
        #         print(f"OUTPUT 1: {output}")
        #         # print(output[1]:output[3], output[0]:output[2])
                
        #         # print(f"MINE!{np.array([output]).shape}")
                
        #         draw_boxes_second(im0s, np.array([output]), [2] , [0], 'Hello')

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                print(im0.shape)
                # if len(det[0])>0:
                #     output = det[0].tolist()[:4]
                #     print(output)
                

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                orig_det = det
                orig_det = orig_det[:, :4].tolist()[0]
                orig_det = [int(x) for x in orig_det]
                                
                
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                output = det[:, :4].tolist()[0]
                output = [int(x) for x in output]
                # draw_boxes_second(im0s, np.array([output]), [2] , [0], 'Hello')
                # print(f"DET!!!!{np.array([output])}")
                
                end_time_det = time.time()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    # print(x1,y1,x2,y2)
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                    
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()

                txt_str = ""

                #*******************************#
                # Second step
                #*******************************#
                # Crop out only the driver side of the detections
                cropped_dets = dets_to_sort.copy()
                cropped_dets[:, 2] = ((cropped_dets[:, 2]-cropped_dets[:, 0]) * 0.7) + cropped_dets[:, 0]
                
                
                for i, det_second in enumerate(cropped_dets[:,:4]):
                    print(f"{i, det_second}")

                    det_second = [int(x) for x in det_second]
                    # print(type(im0))
                    # print(im0.shape)
                    # Padded resize
                    
                    # Crop image
                    cropped_im0 = im0[det_second[1]:det_second[3], det_second[0]:det_second[2]]
                    
                    img_second = letterbox(cropped_im0, imgsz, stride=stride)[0]
                    # print(type(img_second))
                    # print(img_second.shape)
                    # Convert
                    img_second = img_second[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    
                    img_second = np.ascontiguousarray(img_second)
                    img_second = torch.from_numpy(img_second).to(device)
                    img_second = img_second.half() if half else img.float()  # uint8 to fp16/32
                    img_second /= 255.0  # 0 - 255 to 0.0 - 1.0
                    print(f"IMG second {img_second.shape}")
                    # print(img)
                    if img_second.ndimension() == 3:
                        img_second = img_second.unsqueeze(0)
                        
                    pred_second = model_second(img_second, augment=opt.augment)[0]
                    
                    pred_second = non_max_suppression(pred_second, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                    
                    
                    # print(f"BOX!!{det_second}")
                    # draw_boxes_second(im0, det_second)
                    

                # for det_second in dets_to_sort:
                #     print(det_second[:4])

                # #loop over tracks
                # for track in tracks:
                #     # color = compute_color_for_labels(id)
                #     #draw colored tracks
                #     if colored_trk:
                #         [cv2.line(im0, (int(track.centroidarr[i][0]),
                #                     int(track.centroidarr[i][1])), 
                #                     (int(track.centroidarr[i+1][0]),
                #                     int(track.centroidarr[i+1][1])),
                #                     rand_color_list[track.id], thickness=2) 
                #                     for i,_ in  enumerate(track.centroidarr) 
                #                       if i < len(track.centroidarr)-1 ] 
                #     #draw same color tracks
                #     else:
                #         [cv2.line(im0, (int(track.centroidarr[i][0]),
                #                     int(track.centroidarr[i][1])), 
                #                     (int(track.centroidarr[i+1][0]),
                #                     int(track.centroidarr[i+1][1])),
                #                     (255,0,0), thickness=2) 
                #                     for i,_ in  enumerate(track.centroidarr) 
                #                       if i < len(track.centroidarr)-1 ] 
                        
                    
                    
                    # if save_txt:
                    #     # Normalize coordinates
                    #     txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1], track.centroidarr[-1][1] / im0.shape[0])
                    #     if save_bbox_dim:
                    #         txt_str += " %f %f" % (np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0], np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
                    #     txt_str += "\n"
                
                if save_txt:
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(txt_str)

                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names)
                                       
                    
                
                for phone_det in pred_second:
                    if len(phone_det):
                        
                        orig_det = det
                        orig_det = orig_det[:, :4].tolist()[0]
                        orig_det = [int(x) for x in orig_det]
                                        

                        phone_det[:, :4] = scale_coords(img_second.shape[2:], phone_det[:, :4], cropped_im0.shape).round()
                        # output = det[:, :4].tolist()[0]
                        # output = [int(x) for x in output]
                        
                        phone_det = phone_det[:, :4].tolist()[0]
                        phone_det = [int(x) for x in phone_det]
                        print(f'Phone DET: {phone_det}')
                        draw_boxes_second(im0, [[det_second[0]+phone_det[0], det_second[1]+phone_det[1], 
                                                            (phone_det[2]-phone_det[0]) + det_second[0]+phone_det[0],
                                                            (phone_det[3]-phone_det[1])+det_second[1]+phone_det[1]]])
                        

                
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            
            ####ADD FPS###
            if dataset.mode != 'image':
                
                end_time_track = time.time()
                # fps_det = 1/(end_time_det-start_time)
                fps_track = 1/(end_time_track-start_time)
                fps_list_track.append(fps_track)
                
                # Smoth fps
                fps_track = sum(fps_list_track[-30:])/len(fps_list_track[-30:])
                
                # del fps_list_track[:30]
                # start_time = current_time
                # cv2.putText(im0, f"FPS: {str(int(fps_det))}", (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                cv2.putText(im0, f"FPS: {int(fps_track)}", (20,60), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                  cv2.destroyAllWindows()
                  raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--weights_second', nargs='+', type=str, default='', help='model.pt path(s)')
    # parser.add_argument('--download', action='store_true', help='download model weights automatically')
    # parser.add_argument('--no-download', dest='download', action='store_false',help='not download model weights if already exist')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')
    parser.add_argument('--save-bbox-dim', action='store_true', help='save bounding box dimensions with --save-txt tracks')
    
    # parser.set_defaults(download=True)
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    # if opt.download and not os.path.exists(str(opt.weights)):
    #     print('Model weights not found. Attempting to download now...')
    #     download('./')

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
            
