# -*- coding: utf-8 -*-
import os

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import cv2
import numpy as np

from utils.nms import nms

MEAN = np.array([123.675, 116.28, 103.53], np.float32).reshape(1, 1, 3) /255
STD = np.array([58.395, 57.12, 57.375], np.float32).reshape(1, 1, 3) /255


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return 'Host:\n ' + str(self.host) + '\nDevice:\n' + str(self.device)

    def __repr__(self):
        return self.__str__()

class Detector(object):

    def __init__(self, filepath, img_shape):
        self.engine = self.get_engine(filepath)
        self.allocate_buffs(self.engine)
        self.height, self.width = img_shape
        # '道闸','地面垃圾','地锁','减速带','路障','限位器','雪糕筒'
        self.class_names = ('barrier_gate', 'garbage',
               'parking_lock', 'speed_bump', 'road_block', 
               'stopper', 'traffic_cone')
        self.context = self.engine.create_execution_context()

    def get_engine(self, filepath):
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        print("Reading engine from file {}".format(filepath))
        with open(filepath, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffs(self, engine):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        for binding in engine:
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(binding)), dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

    def inference(self, data):
        data = self.pre_process(data)
        self.inputs[0].host = data.ravel()
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()
        output = [out.host for out in self.outputs]
        dets = self.post_process(output)

        return dets

    def pre_process(self, data):
        data = data.astype(np.float32) / 255.
        data = (data - MEAN) / STD
        data = data.transpose(2, 0, 1)

        return data

    def post_process(self, results):
        boxes = results[0].reshape((-1, 4))
        scores = results[1].reshape((-1, 7))
        labels = np.argmax(scores, axis=1)
        scores = np.max(scores, axis=1)
        
        mask = scores > 0.3 
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        keep = nms(boxes, scores, 0.5)
        if len(keep) == 0:
            return ([], [], [])
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return (boxes, scores, labels)
