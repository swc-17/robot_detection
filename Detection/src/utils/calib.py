import cv2
import numpy as np

def get_map():
    calib_file = "utils/calib.yaml"
    cv_file = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)

    camera_matrix = cv_file.getNode("CameraMatrix").mat()
    dist_coef = cv_file.getNode("DistortionCoeffs").mat()
    width = int(cv_file.getNode("Resolution width").real())
    height= int(cv_file.getNode("Resolution height").real())
    img_size = (width, height)
    R = np.eye(3)

    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coef, R, camera_matrix, img_size,  cv2.CV_32FC1)
    return mapx, mapy

map_x, map_y = get_map()
