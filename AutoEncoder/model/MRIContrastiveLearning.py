import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, args, temperature=0.2):
        super(ContrastiveLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, anchor_feats, positive_feats):
        batch_size = anchor_feats.size(0)
        device = anchor_feats.device

        anchor_feats = F.normalize(anchor_feats, p=2, dim=1)
        positive_feats = F.normalize(positive_feats, p=2, dim=1)
        logits = torch.matmul(anchor_feats, positive_feats.T) / self.temperature
        labels = torch.arange(batch_size, device=device)

        loss = F.cross_entropy(logits, labels)
        return loss