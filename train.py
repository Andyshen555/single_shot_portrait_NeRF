import os
import re
import math
import lpips
import torch
import random
import argparse
import numpy as np
from torch_utils import misc
from typing import List, Tuple, Union
from torch.cuda.amp import autocast, GradScaler

import dnnlib
import legacy
from training.model import lp3d
from training.triplane import TriPlaneGenerator
from training.discriminator import Discriminator
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    return samples.unsqueeze(0), voxel_origin, voxel_size

def train(model, G, D, truncation_psi, truncation_cutoff, fov_deg, rank):
    optimizer_lp3d = torch.optim.Adam(model.parameters(), lr=get_learning_rate(0), betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=get_learning_rate(0), betas=(0.9, 0.999))
    loss_l1 = torch.nn.L1Loss()
    loss_bce = torch.nn.BCEWithLogitsLoss()
    loss_lpips = lpips.LPIPS(net='alex').requires_grad_(False).to(device)
    model.train()
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    with torch.autocast(device_type="cuda"):
        true_label = torch.ones(1, 1).to(device)
        false_label = torch.zeros(1, 1).to(device)

    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)

    for epoch in range(args.epoch):
        for i in range(1600):
            lr = get_learning_rate(epoch * 1600 + i)
            # first view
            z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
            angle_p = np.random.uniform(-0.2, 0.2)
            angle_y = np.random.uniform(-0.4, 0.4)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(device)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(device)

            eg_mapping = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            eg_output = G.get_plane(eg_mapping)

            with torch.autocast(device_type="cuda"):
                eg_img = G.synthesis(eg_output, eg_mapping, camera_params)['image']
                lp_output = model(eg_img)
                lp_img = G.synthesis(lp_output, eg_mapping, camera_params)['image']
                # need loss between eg3d and lp3d

            # backward D
            D.requires_grad_(True)
            optimizer_D.zero_grad()
            D_loss = (loss_bce(D(eg_img), true_label)+loss_bce(D(lp_img.detach()), false_label)) * 0.5
            D_loss.backward()

            # backward lp3d
            D.requires_grad_(False)
            optimizer_lp3d.zero_grad()
            lp3d_loss = loss_l1(lp_output, eg_output) + loss_l1(lp_img, eg_img) + 0.1*loss_bce(D(lp_img), true_label) + loss_lpips(lp_img, eg_img)
            lp3d_loss.backward(retain_graph=True)

            #second view
            angle_p = np.random.uniform(-0.2, 0.2)
            angle_y = np.random.uniform(-0.4, 0.4)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            
            with torch.autocast(device_type="cuda"):
                eg_img2 = G.synthesis(eg_output, eg_mapping, camera_params)['image']
                lp_img2 = G.synthesis(lp_output, eg_mapping, camera_params)['image']

            # backward D
            D.requires_grad_(True)
            optimizer_D.zero_grad()
            D_loss = (loss_bce(D(eg_img2), true_label)+loss_bce(D(lp_img2.detach()), false_label)) * 0.5
            D_loss.backward()
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = lr
            optimizer_D.step()

            with torch.autocast(device_type="cuda"):
                # backward lp3d
                D.requires_grad_(False)
                optimizer_lp3d.zero_grad()
                lp3d_loss = 0.025*loss_bce(D(lp_img2), true_label) + loss_l1(lp_img2, eg_img2) + loss_lpips(lp_img2, eg_img2)
            lp3d_loss.backward()
            for param_group in optimizer_lp3d.param_groups:
                param_group['lr'] = lr
            optimizer_lp3d.step()
            print(f'Epoch: {epoch}, Iter: {i}, lp3d_loss: {lp3d_loss.item()}, D_loss: {D_loss.item()}')

        torch.save(model.state_dict(), f'./checkpoint/lp3d.pth')

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_pkl', default='ffhq512-128.pkl', required=False)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='minibatch size')
    parser.add_argument('--truncation_psi', default=1, type=int, help='minibatch size')
    parser.add_argument('--truncation_cutoff', default=14, type=int, help='minibatch size')
    parser.add_argument('--fov_deg', default=18.837, type=int, help='fov degree')
    parser.add_argument('--cont', default=False, type=bool, help='continue training')
    args = parser.parse_args()

    # Initialize torch.distributed
    torch.distributed.init_process_group(backend="nccl")
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(torch.device(local_rank))
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')

    # Set random seed
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = lp3d(True).to(device)
    print("Load model Successfully")
    D = Discriminator().to(device)
    print("Load Discriminator Successfully")
    with dnnlib.util.open_url(args.network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
    print("Reloading Modules!")
    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new
    G.eval().requires_grad_(False).to(device)
    del G_new
    print("Load Teacher Successfully")
    print('Start training...')
    train(model, G, D, args.truncation_psi, args.truncation_cutoff, args.fov_deg, rank=None)