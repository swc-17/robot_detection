# -*- coding: utf-8 -*-
import os 
import sys

import cv2
import numpy as np
import time

from utils.detector import Detector
from utils.visualize import add_2d_detection
from utils.calib import map_x, map_y
from utils.counter import Counter

HEIGHT_RESIZE = 416
WIDTH_RESIZE  = 416
VIDEO = "/media/nvidia/kingston/videos/0802/2022-08-02-10-02-27.avi"
VIS = True
COUNT = True


def main():
    root = '../models/'
    filepath = 'yolo.trt'
    filepath = os.path.join(root, filepath)
    detector = Detector(filepath,(HEIGHT_RESIZE ,WIDTH_RESIZE))

    t = time.strftime("%Y-%m-%d-%H-%M-%S")
    f = cv2.VideoWriter_fourcc('M','J','P','G')
    writer = cv2.VideoWriter(t+'.avi', f, 10.0, (1280, 720), True)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("camera not open")
        exit()
 
    RECORD = False
    while True:        
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (WIDTH_RESIZE, HEIGHT_RESIZE))
        boxes, scores, labels = detector.inference(img)

        img = add_2d_detection(img, boxes, scores, labels)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (1280, 720))
        
        if RECORD:        
            writer.write(img)
        cv2.imshow('detection',img)
        key = cv2.waitKey(1) 
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('s'):
            RECORD = True
            print('start recording video')
        
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

