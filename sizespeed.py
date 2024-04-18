import time
import torch
from training.model import lp3d
from torch.cuda.amp import autocast, GradScaler

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':

    device = torch.device('cuda:0')

    model = lp3d().eval()
    model = model.to(device)
    model = torch.compile(model)
    x = torch.randn(1, 3, 512, 512).to(device)
    y = model(x)
    with (torch.no_grad(), autocast()):
        for i in range(200):
            y = model(x)

        start = time.time()
        for i in range(200):
            y = model(x)
        stop = time.time()
        print("The inference fps on 3090:", 200/(stop - start))