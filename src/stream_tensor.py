import pygame
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from capture import init_gui, draw_image, draw_rectangle_np, clamp_tlbr,draw_text_in_bbox
import pygame

import time

from capture import FrameCapture, YUV422toBGR, YUV422toRGB, openSharedMemory
import cv2
import cv2.cuda
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker

from mmdeploy_runtime import Detector

import torch

classes = (
    'pedestrian',
    'people',
    'bicycle',
    'car',
    'van',
    'truck',
    'tricycle',
    'awning-tricycle',
    'bus',
    'motor',
    'others',
)

class DET_args:
    det_threshold = 0.15
    

class BYTE_args:
    track_thresh = 0.3
    track_buffer = 100
    match_thresh = 0.4
    min_box_area = 10
    mot20 = False
    
track_args = BYTE_args()
det_args = DET_args()

count = 0
total = 0

width, height = 720, 405
width_sr, height_sr = 720, 720
width_paded, height_padded = 736, 416
init_gui(width_paded, height_padded)
frameCapture = FrameCapture()

tracker = BYTETracker(track_args,frame_rate=37)
# detector = Detector(model_path='/home/base/bk/project/models/cascade_final/best_run/deploy/int8', device_name='cuda', device_id=0)
detector = Detector(model_path='/home/base/bk/project/models/yolo_final/best_run/deploy/int8', device_name='cuda', device_id=0)

total_results_ms = []
total_track = []
track = 0
system_time = 0
while True:
    count+=1
    system_time_start = time.perf_counter()
    start_time_system = time.perf_counter()
    
    pygame.event.poll()
    frameCapture.captureRawFrame()
    orig_img_rgb = frameCapture.getRGBFrame()
    ar_image = cv2.resize(orig_img_rgb,(width,height), interpolation = cv2.INTER_AREA)   
    padded_img = cv2.copyMakeBorder(ar_image, 5, 6, 8, 8, cv2.BORDER_CONSTANT, value=(144, 144, 144))
    padded_img = cv2.medianBlur(padded_img, 3) #BEST


    with torch.no_grad():
        bboxes, labels, masks = detector(padded_img)

    if(not len(bboxes)):
        continue
    
    threshold_bboxes = bboxes[bboxes[:, 4] > det_args.det_threshold]

    online_targets = tracker.update(threshold_bboxes, (height, width), (height,width))    
    online_tlwhs = []
    online_ids = []
    online_scores = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        
        vertical = tlwh[2] / tlwh[3] > 1.6
        if tlwh[2] * tlwh[3] > track_args.min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            
            draw_rectangle_np(padded_img, max(0, min(width-1, int(tlwh[0]))), 
                              max(0,min(height,int(tlwh[1]))), 
                              min(width-1, int(tlwh[2])+int(tlwh[0])), 
                              min(height-1,int(tlwh[3])+int(tlwh[1])))
            # draw_text_in_bbox(padded_img, f"ID: {tid} S: {t.score:.2f}" , int(tlwh[0]), int(tlwh[1]), int(tlwh[2])+int(tlwh[0]), int(tlwh[3])+int(tlwh[1]))
            track+=1
            
    # for bbox, label_id in zip(threshold_bboxes, labels):
    #     [xmin, ymin , xmax, ymax], score = bbox[0:4].astype(int), bbox[4]
    #     draw_rectangle_np(padded_img, xmin, ymin, xmax, ymax, color=(255,0,0))
    #     draw_text_in_bbox(padded_img, str(classes[label_id]) + " " + str(f"{score:.2f}"),xmin, ymin, xmax, ymax, color=(255,0,0))
    
    draw_image(padded_img, width_paded, height_padded)
    end_time_system = time.perf_counter()
    total_time_system = end_time_system-start_time_system
    total_results_ms.append(total_time_system*1000)
    total_track.append(track)
    track = 0
    system_time_end = time.perf_counter()
    system_time += (system_time_end-system_time_start)
    print(system_time)
    if(system_time>=300):
        with open("end_results.txt", "w") as result_file:
            result_file.write("time_ms:\n")
            result_file.write("\n ".join(map(str, total_results_ms)))
            result_file.write("tracks:\n")
            result_file.write("\n ".join(map(str, total_track)))
            exit()
            
            

