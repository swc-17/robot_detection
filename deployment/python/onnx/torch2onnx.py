import torch


def saveONNX(model, filepath, c, h, w):
    model = model.cuda()
    dummy_input = torch.randn(1, c, h, w, device='cuda')
    torch.onnx._export(model, dummy_input, filepath, verbose=True)
