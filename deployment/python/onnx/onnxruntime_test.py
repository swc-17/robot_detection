import onnxruntime as ort
import numpy as np


def onnx_test(filepath):
    ort_session = ort.InferenceSession(filepath)
    inp = ort_session.get_inputs()[0]
    input = np.random.random(size=inp.shape)
    outputs = ort_session.run(None, {inp.name:input.astype('float32')})
    print(outputs[0].shape)
    print(outputs[0])


def main():
    filepath = 'model.onnx'
    onnx_test(filepath)


if __name__ == '__main__':
    main()
