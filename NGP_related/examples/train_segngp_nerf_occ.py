"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import pathlib
import time
import matplotlib.pyplot as plt
import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from radiance_fields.ngp_segment import NGPRadianceField

from examples.utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_segoccgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator


def run(args):
    device = "cuda:0"
    set_random_seed(42)

    if args.scene in MIPNERF360_UNBOUNDED_SCENES:
        from datasets.nerf_360_v2 import SubjectLoader

        # training parameters
        max_steps = 50000
        init_batch_size = 1024
        target_sample_batch_size = 1 << 18
        weight_decay = 0.0
        # scene parameters
        aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
        near_plane = 0.0
        far_plane = 1.0e10
        # dataset parameters
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
        test_dataset_kwargs = {"factor": 4}
        # model parameters
        grid_resolution = 128
        grid_nlvl = 4
        # render parameters
        render_step_size = 1e-3
        alpha_thre = 1e-2
        cone_angle = 0.004

    elif args.scene in NERF_SYNTHETIC_SCENES:
        from datasets.nerf_synthetic import SubjectLoader

        # training parameters
        max_steps = args.train_steps
        init_batch_size = 1024
        target_sample_batch_size = 1 << 18
        weight_decay = (
            1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
        )
        # scene parameters
        aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
        near_plane = 0.0
        far_plane = 1.0e10
        # dataset parameters
        train_dataset_kwargs = {}
        test_dataset_kwargs = {}
        # model parameters
        grid_resolution = 128
        grid_nlvl = 1
        # render parameters
        render_step_size = args.render_step_size
        alpha_thre = 0.0
        cone_angle = 0.0

    version = args.version # default: v1

    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split=args.train_split,
        num_rays=init_batch_size,
        device=device,
        **train_dataset_kwargs,
    )

    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split="test",
        num_rays=None,
        device=device,
        **test_dataset_kwargs,
    )

    estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl, version=version
    ).to(device)

    # setup the radiance field we want to train.
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1]).to(device)
    optimizer = torch.optim.Adam(
        radiance_field.parameters(),
        lr=1e-2,
        eps=1e-15,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=100
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 9 // 10,
                ],
                gamma=0.33,
            ),
        ]
    )
    
    reshape_fn = lambda x: x[None, ...].permute(0, 3, 1, 2)
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = structural_similarity_index_measure
    lpips_fn_vgg = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
    lpips_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    # training
    tic = time.time()
    for step in range(max_steps + 1):
        radiance_field.train()
        estimator.train()
        i = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[i]

        render_bkgd = data["color_bkgd"] # white: [1, 1, 1]
        rays = data["rays"] # origins: B 3 viewdirs: B 3
        pixels = data["pixels"] # B 3: gt pixel color

        def occ_eval_fn(x): 
            # x: b 3 -> b 6
            random_vector = torch.randn_like(x).to(x)
            unit_vector = random_vector / torch.norm(random_vector, dim=-1, keepdim=True) # b 1
            x_start = x - unit_vector * 0.5 * render_step_size
            x_end = x + unit_vector * 0.5 * render_step_size
            x_new = torch.cat([x_start, x_end], dim=-1) # b 6
            density = radiance_field.query_density(x_new)
            return density * render_step_size

        # update occupancy grid
        estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

        # render
        rgb, acc, depth, n_rendering_samples = render_image_with_segoccgrid(
            radiance_field,
            estimator,
            rays,
            # rendering options
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
            version=version
        )
        if n_rendering_samples == 0:
            continue

        if target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)

        # compute loss
        loss = F.smooth_l1_loss(rgb, pixels)

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()

        if step % 1000 == 0 and False:
            elapsed_time = time.time() - tic
            loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(loss) / np.log(10.0)
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss:.5f} | psnr={psnr:.2f} | "
                f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                f"max_depth={depth.max():.3f} | "
            )

        if step > 0 and step % (max_steps // 10 - 1) == 0:
            # evaluation
            radiance_field.eval()
            estimator.eval()

            psnrs = []
            lpips_vgg = []
            lpips_alex = []
            ssim = []
            with torch.no_grad():
                # for i in tqdm.tqdm(range(len(test_dataset))):
                for i in range(len(test_dataset)):
                    data = test_dataset[i]
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    pixels = data["pixels"]

                    rgb, acc, depth, _ = render_image_with_segoccgrid(
                        radiance_field,
                        estimator,
                        rays,
                        # rendering options
                        near_plane=near_plane,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=cone_angle,
                        alpha_thre=alpha_thre,
                        version=version
                    )
                    
                    rgb = torch.clip(rgb, min=0.0, max=1.0)
                    pixels = torch.clip(pixels, min=0.0, max=1.0)
                    psnrs.append(psnr_fn(reshape_fn(rgb), reshape_fn(pixels)).item())
                    lpips_vgg.append(lpips_fn_vgg(reshape_fn(rgb), reshape_fn(pixels)).item())
                    lpips_alex.append(lpips_fn_alex(reshape_fn(rgb), reshape_fn(pixels)).item())
                    ssim.append(ssim_fn(reshape_fn(rgb), reshape_fn(pixels)).item())
                    
            psnr_avg = sum(psnrs) / len(psnrs)
            lpips_vgg_avg = sum(lpips_vgg) / len(lpips_vgg)
            lpips_alex_avg = sum(lpips_alex) / len(lpips_alex)
            ssim_avg = sum(ssim) / len(ssim)
            print(f"step {step}: evaluation: psnr_avg={psnr_avg}, lpips_vgg_avg={lpips_vgg_avg}, lpips_alex_avg={lpips_alex_avg}, ssim_avg={ssim_avg}")
            
            ckpts_save_path = f"./ckpts/{args.exp_time}/seg/{args.scene}"

            if not os.path.exists(ckpts_save_path):
                os.makedirs(ckpts_save_path)

            torch.save(radiance_field.state_dict(), os.path.join(ckpts_save_path, 'radiance_field.pt'))
            torch.save(estimator.state_dict(), os.path.join(ckpts_save_path, 'estimator.pt'))

            print(f'Models saved to {ckpts_save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        # default=str(pathlib.Path.cwd() / "data/360_v2"),
        default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
        help="the root dir of the dataset",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
        help="which scene to use",
    )
    parser.add_argument(
        "--exp_time",
        type=str,
        default="dev_test"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1"
    )
    parser.add_argument(
        "--render_step_size",
        type=float,
        default=5e-3
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=50000
    )

    args = parser.parse_args()

    run(args)
