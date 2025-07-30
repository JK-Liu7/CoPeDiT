import argparse
import os
import timeit
from copy import deepcopy
from datetime import datetime
from time import time
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from torch.nn import L1Loss, MSELoss
from timm.scheduler.cosine_lr import CosineLRScheduler
from monai.utils import first

from AutoEncoder.model.CoPeVAE_BrainMRI import CopeVAE
from LDM.model.MDiT3D_Brain import *
from LDM.dataset.BrainMRI_data import *

from torch.cuda.amp import GradScaler, autocast
from LDM.inferers import inferer_DiT_Brain
from LDM.utils import *
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler


def train(args, autoencoder, DiT):
    if args.rank == 0:
        logger = create_logger(args.log_dir, args.distributed)
    else:
        logger = create_logger(args.log_dir, args.distributed)

    keys = {
        "BraTS": ["t1", "t2", "t1ce", "flair"],
        "IXI": ["t1", "t2", "pd"]
    }

    datasets = get_brainMRI(args)
    train_files_combined, test_files_combined = get_data(datasets, args.dataset)
    train_transform, test_transform = get_transforms(args, keys[args.dataset])
    dataloader_train, dataloader_test, train_sampler, test_sampler = get_loader(args, args.rank, args.world_size, train_files_combined,test_files_combined, train_transform, test_transform)

    if args.diff_loss == "l2":
        diff_loss = MSELoss()
        if args.rank == 0:
            print("Use l2 loss")
    else:
        diff_loss = L1Loss(reduction="mean")
        if args.rank == 0:
            print("Use l1 loss")

    accumulation_steps = args.gradient_accumulation_steps

    if args.noise_scheduler == 'linear':
        scheduler_ddpm = DDPMScheduler(num_train_timesteps=args.diffusion_steps, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195, clip_sample=False)
    elif args.noise_scheduler == 'cosine':
        scheduler_ddpm = DDPMScheduler(num_train_timesteps=args.diffusion_steps, schedule="cosine", clip_sample=False)

    ema = deepcopy(DiT).to(args.device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    DiT = DiT.to(args.device)

    # load autoencoder checkpoint
    autoencoder = autoencoder.to(args.device)
    state_dict = args.ae_dir + 'vqvae/Autoencoder.pt'

    if not state_dict is None:
        state_dict = torch.load(str(state_dict))
        autoencoder.load_state_dict(state_dict["model"])
        if args.rank == 0:
            print('AE weighted loaded!')
    autoencoder.eval()
    # Freeze vae and prompt_encoder
    autoencoder.requires_grad_(False)

    check_data = first(dataloader_train)
    with torch.no_grad():
        with autocast(enabled=True):
            z0 = autoencoder.encode_stage_2_inputs(check_data[0].to(args.device))
            z1 = autoencoder.encode_stage_2_inputs(check_data[1].to(args.device))
    scale_factor = 1.0
    print(f"scale_factor -> {scale_factor}.")
    print(z0.shape)
    print(z1.shape)
    torch.cuda.empty_cache()

    # define infer
    inferer = inferer_DiT_Brain.LatentDiffusionInferer(scheduler_ddpm, scale_factor=scale_factor)

    logger.info(f"AE Parameters: {sum(p.numel() for p in autoencoder.parameters()):,}")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in DiT.parameters()):,}")

    if args.opt == "adam":
        optimizer = optim.Adam(params=DiT.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=DiT.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(params=DiT.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.lr_schedule == "warmup_cosine":
        scheduler = CosineLRScheduler(optimizer, warmup_t=args.warmup_steps, warmup_lr_init=1e-6, t_initial=args.num_steps, lr_min=args.lr_min, cycle_limit=1)
    elif args.lr_schedule == "poly":
        def lambdas(epoch):
            return (1 - float(epoch) / float(args.epochs)) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    if args.amp:
        scaler = GradScaler()

    DiT.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    val_interval = args.val_interval
    start_epoch = 0
    max_epochs = args.epochs

    start = timeit.default_timer()

    if args.distributed:
        DiT = DistributedDataParallel(DiT, device_ids=[args.rank])

    # resume:
    if args.resume_ckpt is not None:
        ckpt = torch.load(str(args.resume_ckpt), map_location='cpu')
        DiT.module.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"]
        logger.info(f"Resuming training from checkpoint, epoch: {start_epoch}")

    if args.resume_ckpt is None:
        if args.distributed:
            update_ema(ema, DiT.module, decay=0)  # Ensure EMA is initialized with synced weights
        else:
            update_ema(ema, DiT, decay=0)

    for epoch in range(start_epoch, max_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

        DiT.train()
        train_epoch_losses = {"diff_loss": 0}
        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), ncols=150)
        progress_bar.set_description(f"Epoch {epoch}")

        for i, batch in progress_bar:
            x_available, x_missing, missing_condition = batch
            x_available, x_missing, missing_condition = (x_available.to(args.device), x_missing.to(args.device), missing_condition.to(args.device))
            with torch.no_grad():
                with autocast(enabled=args.amp):
                    prompts = autoencoder.get_condition(x_available)

            with autocast(enabled=args.amp):
                noise, noise_missing = get_noise(z0, z1)
                noise, noise_missing = noise.to(args.device), noise_missing.to(args.device)
                # Create timesteps
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (x_available.shape[0],), device=x_available.device).long()

                # Get model prediction
                pred, latent = inferer(
                    inputs=[x_available, x_missing], autoencoder_model=autoencoder, diffusion_model=DiT,
                    noise=noise_missing,
                    timesteps=timesteps,
                    condition=prompts
                )
                if args.pred_type == 'noise':
                    l_diff = diff_loss(pred, noise_missing)
                elif args.pred_type == 'x':
                    l_diff = diff_loss(pred, latent)
                elif args.pred_type == 'v':
                    v = scheduler_ddpm.get_velocity(latent, noise, timesteps)
                    l_diff = diff_loss(pred, v)

                l_diff = l_diff / accumulation_steps
                losses = {
                    'diff_loss': l_diff * accumulation_steps
                }

            for loss_name, loss_value in losses.items():
                train_epoch_losses[loss_name] += loss_value.item()

            if args.amp:
                scaler.scale(l_diff).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(DiT.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                l_diff.backward()
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(DiT.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if args.distributed:
                update_ema(ema, DiT.module)
            else:
                update_ema(ema, DiT)
        if args.lrdecay:
            scheduler.step(epoch)

        for key in train_epoch_losses:
            train_epoch_losses[key] /= len(dataloader_train)
        diff_loss_epoch = train_epoch_losses["diff_loss"]

        end = timeit.default_timer()
        time = end - start

        if args.rank == 0:
            print(
                "[Train Epoch: %d][Time: %d][L: %.4f][lr: %.6f]"
                % (epoch, time, diff_loss_epoch, scheduler._get_lr(epoch)[0])
            )

        logger.info(
            "[Train Epoch: %d][Time: %d][L: %.4f][lr: %.6f]"
            % (epoch, time, diff_loss_epoch, scheduler._get_lr(epoch)[0])
        )

        if args.distributed:
            if args.rank == 0:
                checkpoint = {
                    "model": DiT.module.state_dict(),
                    "ema": ema.state_dict(),
                    "args": args,
                    "epoch": epoch
                }
                checkpoint_path = args.model_dir + f"DiT.pt"
                torch.save(checkpoint, checkpoint_path)
        else:
            checkpoint = {
                "model": DiT.state_dict(),
                "ema": ema.state_dict(),
                "args": args,
                "epoch": epoch
            }
            checkpoint_path = args.model_dir + f"DiT.pt"
            torch.save(checkpoint, checkpoint_path)

        if epoch % args.ckpt_interval == 0 and epoch > 0:
            if args.distributed:
                if args.rank == 0:
                    checkpoint = {
                        "model": DiT.module.state_dict(),
                        "ema": ema.state_dict(),
                        "args": args,
                        "epoch": epoch
                    }
                    checkpoint_path = args.model_dir + f"DiT_epoch{epoch}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            else:
                checkpoint = {
                    "model": DiT.state_dict(),
                    "ema": ema.state_dict(),
                    "args": args,
                    "epoch": epoch
                }
                checkpoint_path = args.model_dir + f"DiT_epoch{epoch}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Inference
        if epoch % val_interval == 0 and epoch > 0:
            DiT.eval()  # important! This disables randomized embedding dropout
            infer_epoch_losses = {"diff_loss": 0}
        
            for batch in dataloader_test:
                with torch.no_grad():
                    with autocast(enabled=args.amp):
                        x_available, x_missing, missing_condition = batch
                        x_available, x_missing, missing_condition = (x_available.to(args.device), x_missing.to(args.device), missing_condition.to(args.device))
                        prompts = autoencoder.get_condition(x_available)

                        noise, noise_missing = get_noise(z0, z1)
                        noise, noise_missing = noise.to(args.device), noise_missing.to(args.device)

                        # Create timesteps
                        timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (x_available.shape[0],), device=x_available.device).long()
        
                        # Get model prediction
                        pred, latent = inferer(
                            inputs=[x_available, x_missing], autoencoder_model=autoencoder, diffusion_model=DiT,
                            noise=noise_missing,
                            timesteps=timesteps,
                            condition=prompts
                        )
                        if args.pred_type == 'noise':
                            l_diff = diff_loss(pred, noise_missing)
                        elif args.pred_type == 'x':
                            l_diff = diff_loss(pred, latent)
                        elif args.pred_type == 'v':
                            v = scheduler_ddpm.get_velocity(latent, noise, timesteps)
                            l_diff = diff_loss(pred, v)
                        losses = {
                            'diff_loss': l_diff
                        }

                    for loss_name, loss_value in losses.items():
                        infer_epoch_losses[loss_name] += loss_value.item()
        
            for key in infer_epoch_losses:
                infer_epoch_losses[key] /= len(dataloader_test)
            diff_loss_epoch = infer_epoch_losses["diff_loss"]

            end = timeit.default_timer()
            time = end - start
    
            if args.rank == 0:
                print(
                    "[Infer Epoch: %d][Time: %d][L: %.4f]"
                    % (epoch, time, diff_loss_epoch)
                )
    
            logger.info(
                    "[Infer Epoch: %d][Time: %d][L: %.4f]"
                    % (epoch, time, diff_loss_epoch)
                )

    if args.distributed:
        cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='BraTS', type=str)
    parser.add_argument("--missing_num", default=1, type=int, help="number of missing modalities", choices=[1, 2, 3])
    parser.add_argument("--epochs", default=20000, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=8000, type=int, help="number of training iterations")
    parser.add_argument("--warmup_steps", default=50, type=int, help="warmup steps")
    parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    parser.add_argument("--lrdecay", default=True, help="enable learning rate decay")
    parser.add_argument("--decay", default=0, type=float, help="decay rate")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--lr_min", default=2e-5, type=float)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--val_interval", default=50, type=int)
    parser.add_argument("--ckpt_interval", default=400, type=int)

    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--cache', default=0.2, type=float)
    parser.add_argument('--distributed', default=True, action='store_true', help='distributed training')
    parser.add_argument("--gpu_ids", default=[0, 1, 2, 3], help="local rank")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int)

    # MDiT3D Configuration
    parser.add_argument("--DiT", default='MDiT3D-B/2', type=str)
    parser.add_argument("--noise_scheduler", default='linear', type=str)
    parser.add_argument("--pred_type", default='x', type=str, choices=['noise', 'x', 'v'])
    parser.add_argument('--diffusion_steps', default=500, type=int)
    parser.add_argument('--sample_steps', default=200, type=int)
    parser.add_argument("--diff_loss", default='l2', type=str)

    # CoPeVAE Configuration
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--vae_channel", default=(256, 384, 512), type=Sequence[int])
    parser.add_argument("--res_channel", default=256, type=Sequence[int])
    parser.add_argument('--code_num', default=8192, type=int)
    parser.add_argument('--code_dim', default=8, type=int)
    parser.add_argument("--recon_loss", default='l1', type=str)
    parser.add_argument("--brain_pad", default=(240, 240, 128), type=Sequence[int])
    parser.add_argument("--brain_roi", default=(192, 192, 80), type=Sequence[int])
    parser.add_argument("--brain_size", default=(192, 192, 64), type=Sequence[int])
    parser.add_argument('--modality_num', default=4, type=int)
    parser.add_argument('--latent_dim', default=8, type=int)
    parser.add_argument('--proj_dim', default=512, type=int)
    parser.add_argument('--contrast_dim', default=128, type=int)
    parser.add_argument('--lambda1', default=1.0, type=float)
    parser.add_argument('--lambda2', default=0.1, type=float)
    parser.add_argument('--lambda3', default=1.0, type=float)
    parser.add_argument('--vq_weight', default=1.0, type=float)
    parser.add_argument('--per_weight', default=0.01, type=float)
    parser.add_argument('--adv_weight', default=0.01, type=float)
    parser.add_argument('--pretext_weight', default=0.1, type=float)

    args = parser.parse_args()

    today_date = datetime.today().strftime('%Y.%m.%d')
    args.result_dir = '../result/MDiT3D/'
    os.makedirs(args.result_dir, exist_ok=True)
    args.log_dir = args.result_dir + '/log/Brain/' + args.dataset + '/' + str(args.missing_num) + '/'

    os.makedirs(args.log_dir, exist_ok=True)

    args.ae_dir = 'model_save/CoPeVAE/Brain/' + str(args.dataset) + '/'
    args.model_dir = 'model_save/MDiT3D/Brain/' + args.dataset + '/' + str(args.missing_num) + '/' + today_date + '/'
    os.makedirs(args.model_dir, exist_ok=True)

    args.resume_ckpt = None

    brain_roi = {
        'BraTS': (192, 192, 80),
        'IXI': (240, 240, 90)
    }
    args.brain_roi = brain_roi[args.dataset]
    modality_num = {
        'BraTS': 4,
        'IXI': 3
    }
    args.modality_num = modality_num[args.dataset]

    args.amp = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    args.distributed = False

    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.world_size = 1
    args.rank = 0

    if args.distributed:
        dist.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.device = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.device)
        num_gpus = torch.cuda.device_count()
        print(f"Setting up distributed training with {num_gpus} GPUs available")
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        args.device = torch.device("cuda:0")
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    autoencoder = CopeVAE(args)
    DiT = MDiT3D_models[args.DiT]()

    train(args, autoencoder, DiT)

