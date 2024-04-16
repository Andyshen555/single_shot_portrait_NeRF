import torch
from training.model import lp3d

input = torch.ones((1, 3, 512, 512)).to('cuda:0')
model = lp3d().to('cuda:0')
output = model(input)
torch.save(model.state_dict(), 'temp.pth')