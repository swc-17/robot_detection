import os 
import sys

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


def build_engine(filepath):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        if builder.platform_has_fast_fp16:
            print('this card support fp16')
        if builder.platform_has_fast_int8:
            print('this card support int8')

        with open(filepath, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        with builder.create_builder_config() as config:
            config.max_workspace_size = 1 << 20 
            with builder.build_engine(network, config) as engine:
                serialized_engine = engine.serialize()
                with open('model.engine', 'wb') as f:
                    f.write(engine.serialize())
                print('build engine,saved to model.engine')
                return engine


def main():
    filepath = 'yolo_coco.onnx'
    filepath = os.path.join(sys.path[0], filepath)
    build_engine(filepath)


if __name__ == '__main__':
    main()
