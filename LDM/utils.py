import os
from datetime import datetime
from time import time
import logging
import numpy as np
import torch
import torch.distributed as dist
import subprocess
from collections import OrderedDict


def cleanup():
    dist.destroy_process_group()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(log_dir, distributed):
    """
    Create a logger that writes to a log file and stdout.
    """
    today_date = datetime.today().strftime('%Y.%m.%d')
    if distributed:
        if dist.get_rank() == 0:  # real logger
            logging.basicConfig(filename=log_dir + f"{today_date}.log",
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
            logger = logging.getLogger(__name__)
        else:  # dummy logger (does nothing)
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
    else:
        logging.basicConfig(filename=log_dir + f"{today_date}.log",
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        logger = logging.getLogger(__name__)

    return logger


def calculate_scale_factor(z):
    scale_factor = 1 / torch.std(z)
    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    return scale_factor


def get_noise(z_a, z_m):
    noise_available = torch.zeros_like(z_a)
    noise_missing = torch.randn_like(z_m)
    noise = torch.cat([noise_available, noise_missing], dim=1)
    return noise, noise_missing



