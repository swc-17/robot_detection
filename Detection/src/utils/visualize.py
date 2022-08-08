import cv2
import numpy as np


def add_2d_bbox(img, bbox, score, label, show_txt=True): 
    bbox = np.array(bbox, dtype=np.int32)
    color = PALETTE[label]

    txt = '{}{:.1f}'.format(names[label], score)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.3, 2)[0]
    cv2.rectangle(
      img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    if show_txt:
      cv2.rectangle(img,
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2), color, -1)
      cv2.putText(img, txt, (bbox[0], bbox[1] - 2), 
                  font, 0.3, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return img


def add_2d_detection(img, boxes, scores, labels, show_txt=True, center_thresh=0.):
    for box, score, label in zip(boxes, scores, labels):
        add_2d_bbox(img, box, score, label, show_txt)
    return img


PALETTE = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
            (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
            (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252),
            (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0),
            (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]
names = ('barrier_gate', 'garbage', 'ground_damage',
               'parking_lock', 'speed_bump', 'road_block', 
               'stopper', 'traffic_cone')

