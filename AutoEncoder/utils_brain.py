import os
from datetime import datetime
from time import time
import logging
import numpy as np
import torch
import torch.distributed as dist
from collections import OrderedDict
from einops import rearrange


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


def get_train_loss(x_in, x_rec, intensity_loss, loss_perceptual):
    loss1 = intensity_loss(x_rec['x_incomp'], x_in['x_incomp']) + intensity_loss(x_rec['x_missing'], x_in['x_missing'])
    loss2 = loss_perceptual(x_rec['x_incomp'], x_in['x_incomp']) + loss_perceptual(x_rec['x_missing'], x_in['x_missing'])
    return loss1 / 2, loss2 / 2

def get_adv_loss(x_rec, discriminator, adv_loss):
    x_rec0 = torch.cat((x_rec['x_incomp'], x_rec['x_missing']), dim=0)
    logits_fake = discriminator(x_rec0.contiguous().float())[-1]
    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
    return generator_loss

def get_discriminator_loss(x_in, x_rec, discriminator, adv_loss):
    x_in0 = torch.cat((x_in['x_incomp'], x_in['x_missing']), dim=0)
    x_rec0 = torch.cat((x_rec['x_incomp'], x_rec['x_missing']), dim=0)
    logits_fake = discriminator(x_rec0.contiguous().detach())[-1]
    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
    logits_real = discriminator(x_in0.contiguous().detach())[-1]
    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
    loss_d = (loss_d_fake + loss_d_real) * 0.5
    return loss_d

def train_loss_weighted_sum(args, losses):
    return (losses["rec_loss"] + args.vq_weight * losses["vq_loss"] + args.per_weight * losses["per_loss"]
    + args.adv_weight * losses["adv_loss"] + args.pretext_weight * losses["pretext_loss"])

def test_loss_weighted_sum(args, losses):
    return losses["rec_loss"] + args.per_weight * losses["per_loss"]

def pretext_loss_weighted_sum(args, losses):
    return args.lambda1 * losses["l_len"] + args.lambda2 * losses["l_loc"] + args.lambda3 * losses["l_con"]