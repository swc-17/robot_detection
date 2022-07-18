import os 
import sys

import rospy
from sensor_msgs.msg import Image
import cv_bridge
import cv2
import numpy as np
import ros_numpy

from utils.detector import Detector
from utils.decode import decode
from ros_visualize.msg import detected_object, detected_object_array


HEIGHT_RESIZE = 1088
WIDTH_RESIZE  = 1440

def main():
    root = '/home/swc/catkin_ws/src/deployment/model'
    filepath = 'model-fp32-pix.trt'
    filepath = os.path.join(root, filepath)
    detector = Detector(filepath,(HEIGHT_RESIZE ,WIDTH_RESIZE))

    rospy.init_node('inference', anonymous=True)
    pub = rospy.Publisher('vision_objects_array', detected_object_array, queue_size=10)
    bridge = cv_bridge.CvBridge()
    rospy.sleep(2)	

    while(True):
        msg = rospy.wait_for_message("/image_raw", Image)
        img = ros_numpy.numpify(msg)
        data = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
        data = cv2.resize(data, (WIDTH_RESIZE, HEIGHT_RESIZE))
        results = detector.inference(data)

        det_obj_array = detected_object_array()
        for i in range(1,4):
            for obj in results[i]:
                det_obj = detected_object()
                det_obj.type = detector.class_names[i - 1]
                det_obj.bbox2d = obj[1:5]
                # det_obj.dimension = box[3:6].numpy()
                # det_obj.location = box[:3].numpy()
                # det_obj.rotation = box[6].numpy()
                det_obj.score = obj[12]
                det_obj_array.objects.append(det_obj)

        pub.publish(det_obj_array)
    

if __name__ == '__main__':
    main()


# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#       with open("/home/swc/CenterNet/src/model-fp32.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
# 		      engine = runtime.deserialize_cuda_engine(f.read())
#       h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
#       h_output = [cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(i)), dtype=np.float32) for i in range(1,7)]
#       # Allocate device memory for inputs and outputs.
#       d_input = cuda.mem_alloc(h_input.nbytes)
#       d_output = [cuda.mem_alloc(h_output[i].nbytes) for i in range(6)]
#       # Create a stream in which to copy inputs/outputs and run inference.
#       stream = cuda.Stream()
#       bindings = [int(d_output[i]) for i in range(6)]
#       bindings.insert(0,int(d_input))

#       with engine.create_execution_context() as context:
#           # Transfer input data to the GPU.
#           cuda.memcpy_htod_async(d_input, h_input, stream)
#           # Run inference.
#           context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#           # Transfer predictions back from the GPU.
#           [cuda.memcpy_dtoh_async(h_output[i], d_output[i], stream) for i in range(6)]
#           # Synchronize the stream
#           stream.synchronize()
