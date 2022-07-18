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


def add_3d_detection(
self, image_or_path, dets, calib, show_txt=False, 
center_thresh=0.5, img_id='det'):
    if isinstance(image_or_path, np.ndarray):
        self.imgs[img_id] = image_or_path
    else: 
        self.imgs[img_id] = cv2.imread(image_or_path)
    for cat in dets:
        for i in range(len(dets[cat])):
            cl = (self.colors[cat - 1, 0, 0]).tolist()
            if dets[cat][i, -1] > center_thresh:
                dim = dets[cat][i, 5:8]
                loc  = dets[cat][i, 8:11]
                rot_y = dets[cat][i, 11]
                # loc[1] = loc[1] - dim[0] / 2 + dim[0] / 2 / self.dim_scale
                # dim = dim / self.dim_scale
                if loc[2] > 1:
                    box_3d = compute_box_3d(dim, loc, rot_y)
                    box_2d = project_to_image(box_3d, calib)
                    self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)

PALETTE = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
            (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
            (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252),
            (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0),
            (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]

color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
names = ('barrier_gate', 'garbage', 'ground_damage',
               'parking_lock', 'speed_bump', 'road_block', 
               'stopper', 'traffic_cone')