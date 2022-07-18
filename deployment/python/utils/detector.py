import os

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import cv2
import numpy as np
import torch

from utils.decode import decode
from utils.post_process import post_process
import time 

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
        # '道闸','地面垃圾','地面破损及凹坑','地锁','减速带','路障','限位器','雪糕筒'
        self.class_names = ('barrier_gate', 'garbage', 'ground_damage',
               'parking_lock', 'speed_bump', 'road_block', 
               'stopper', 'traffic_cone')

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
            print(binding)
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
        s = time.time()

        with self.engine.create_execution_context() as context:
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
            context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
            self.stream.synchronize()
        e = time.time()
        print(e - s)
        output = [out.host for out in self.outputs]
        # dets = self.post_process(output)

        return output

    def pre_process(self, data):
        data = data.astype(np.float32) / 255.
        data = (data - MEAN) / STD
        data = data.transpose(2, 0, 1)
        
        return data

    def post_process(self, h_output):
        output = {}
        output["hm"] = torch.from_numpy(h_output[0]).reshape(3, int(self.height / 4), int(self.width / 4)).unsqueeze(0)
        output["dep"] = torch.from_numpy(h_output[1]).reshape(1, int(self.height / 4), int(self.width / 4)).unsqueeze(0)
        output["rot"] = torch.from_numpy(h_output[2]).reshape(8, int(self.height / 4), int(self.width / 4)).unsqueeze(0)
        output["dim"] = torch.from_numpy(h_output[3]).reshape(3, int(self.height / 4), int(self.width / 4)).unsqueeze(0) 
        output["wh"] = torch.from_numpy(h_output[4]).reshape(2, int(self.height / 4), int(self.width / 4)).unsqueeze(0) 
        output["reg"] = torch.from_numpy(h_output[5]).reshape(2, int(self.height / 4), int(self.width / 4)).unsqueeze(0) 
        output['hm'] = output['hm'].sigmoid_()
        output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        
        dets = decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], output['wh'], output['reg'])
        dets = dets.detach().cpu().numpy()
        results = post_process(dets, self.height, self.width)
        return results

        
