import os
import re
import math
import wandb
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
        return args.init_lr * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (args.init_lr - 1e-6) * mul + 1e-6


def train(model, G, D, truncation_psi, truncation_cutoff, fov_deg):
    optimizer_lp3d = torch.optim.Adam([
        {'params': model.deeplabv3.parameters(), 'lr': 1e-4},
        {'params': model.conv1.parameters(), 'lr': 1e-4},
        {'params': model.encoder_high.parameters(), 'lr': 1e-4},
        {'params': model.vit1.parameters(), 'lr': 5e-5},
        {'params': model.vit2.parameters(), 'lr': 5e-5},
    ])
    optimizer_D = torch.optim.Adam(D.parameters(), lr=5e-5)
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
        for i in range(args.step_per_epoch):
            cur_step = epoch * args.step_per_epoch + i
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

            if cur_step > 30000:
                # backward D
                D.requires_grad_(True)
                optimizer_D.zero_grad()
                D_loss = (loss_bce(D(eg_img), true_label)+loss_bce(D(lp_img.detach()), false_label)) * 0.5
                D_loss.backward()
                optimizer_D.step()

                D.requires_grad_(False)
                loss_adv = loss_bce(D(lp_img), true_label)
                if i % 10 == 1 and rank == 0:
                    wandb.log({"loss/adv": loss_adv.item()})

            # backward lp3d
            optimizer_lp3d.zero_grad()

            loss_model = loss_l1(lp_output, eg_output)
            loss_img = loss_l1(lp_img, eg_img)
            loss_lp = loss_lpips(lp_img, eg_img)

            lp3d_loss =  0.01*loss_model + loss_img + loss_lp

            if cur_step > 30000:
                lp3d_loss = lp3d_loss + loss_adv

            lp3d_loss.backward()
            optimizer_lp3d.step()

            if i % 10 == 1 and rank == 0:
                wandb.log({"learning_rate": 1e-4, "loss/model_l1":loss_model.item(), "loss/img_l1": loss_img.item(), "loss/lpips": loss_lp.item()})

        torch.save(model.state_dict(), f'./checkpoint/lp3d.pth')

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_pkl', default='ffhq512-128.pkl', required=False)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--step_per_epoch', default=1600, type=int)
    parser.add_argument('--init_lr', default=2e-4, type=int)
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

    if(rank == 0):
        wandb.init(
            # set the wandb project where this run will be logged
            project="lp3d",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.init_lr,
            "architecture": "lp3d",
            "dataset": "run time generation from eg3d",
            "epochs": 300,
            "note": "None"
            }
        )

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
        G = legacy.load_network_pkl(f)['G_ema']

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
    train(model, G, D, args.truncation_psi, args.truncation_cutoff, args.fov_deg)