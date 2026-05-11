import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

from monai.networks.nets.vqvae import VQVAE
from AutoEncoder.model.MRIContrastiveLearning import ContrastiveLoss


class projection_head(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=512, out_dim=512):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_dim // 2, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv3d(in_channels=hidden_dim // 2, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_dim, affine=True),
            nn.ReLU(inplace=False)
        )

        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim * 2, out_dim)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.conv(input)
        b = x.size()[0]
        x_proj = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(b, -1)
        x_out = self.layer(x_proj)
        return x_out

class similarity_head(nn.Module):
    def __init__(self, args, in_dim=512, out_dim=2):
        super().__init__()
        self.args = args
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.in_dim, self.out_dim),
        )
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x_proj, label):
        b = x_proj.size()[0]
        output = self.mlp(x_proj)
        loss = self.loss(output, label)
        return loss

class contrastive_head(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.in_dim, self.out_dim),
        )

    def forward(self, x_proj):
        output = self.projection(x_proj)
        return output


class CopeVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ae = VQVAE(spatial_dims=self.args.spatial_dims, in_channels=1, out_channels=1, num_res_channels=self.args.res_channel,
                        num_embeddings=self.args.code_num,
                        embedding_dim=self.args.code_dim,
                        channels=self.args.vae_channel,
                        downsample_parameters=(
                            (2, 4, 1, 1),
                            (2, 4, 1, 1),
                            (2, 4, 1, 1),
                        ),
                        upsample_parameters=(
                            (2, 4, 1, 1, 0),
                            (2, 4, 1, 1, 0),
                            (2, 4, 1, 1, 0),
                        )
                        )

        self.proj_head1 = projection_head(in_dim=self.args.latent_dim, hidden_dim=self.args.proj_dim // 2, out_dim=self.args.proj_dim)
        self.proj_head2 = projection_head(in_dim=self.args.latent_dim, hidden_dim=self.args.proj_dim // 2, out_dim=self.args.proj_dim)
        self.proj_head3 = projection_head(in_dim=self.args.latent_dim, hidden_dim=self.args.proj_dim // 2, out_dim=self.args.proj_dim)

        self.contrastive_head = contrastive_head(in_dim=self.args.proj_dim, out_dim=self.args.contrast_dim)
        self.classifier_length = similarity_head(args=self.args, in_dim=self.args.proj_dim, out_dim=3)
        self.classifier_positioning = similarity_head(args=self.args, in_dim=self.args.proj_dim, out_dim=self.args.length)

        self.cl = ContrastiveLoss(self.args)


    def forward(self, x_incomp, x_missing, missing_length, missing_label):

        incomp_emb = self.encode_proj(x_incomp)
        missing_emb = self.encode_proj(x_missing)

        l_loc = self.incompleteness_positioning(incomp_emb, missing_label)
        l_len = self.missing_length_detection(incomp_emb, missing_length)
        l_con = self.missing_slice_assessment(incomp_emb, missing_emb)

        incomp_rec, incomp_qloss = self.decode(incomp_emb)
        missing_rec, missing_qloss = self.decode(missing_emb)
        l_vq = incomp_qloss + missing_qloss

        x_in = {'x_incomp': x_incomp,
                 'x_missing': x_missing
        }

        x_rec = {'x_incomp': incomp_rec,
                 'x_missing': missing_rec
        }

        l_pretext = {'l_len': l_len,
                     'l_loc': l_loc,
                     'l_con': l_con
        }
        return x_in, x_rec, l_vq, l_pretext

    def val(self, x_in):
        z = self.ae.encode(x_in)
        x_emb, _ = self.ae.quantize(z)
        x_rec = self.ae.decode(x_emb)
        return x_rec

    def get_condition(self, x_in):
        x_proj = self.encode_proj(x_in)
        len_proj = self.proj_head1(x_proj)
        loc_proj = self.proj_head2(x_proj)
        con_proj = self.proj_head3(x_proj)
        out_condition = torch.stack([len_proj, loc_proj, con_proj], dim=1)
        return out_condition


    def missing_length_detection(self, x_incomp, missing_length):
        x_incomp_out = self.proj_head1(x_incomp)
        l_len = self.classifier_length(x_incomp_out, missing_length)
        return l_len

    def incompleteness_positioning(self, x_incomp, missing_idx):
        x_incomp_out = self.proj_head2(x_incomp)
        l_loc_out = self.classifier_positioning(x_incomp_out, missing_idx)
        return l_loc_out

    def missing_slice_assessment(self, x_incomp, x_missing):
        x_incomp_con = self.contrastive_head(self.proj_head3(x_incomp))
        x_missing_con = self.contrastive_head(self.proj_head3(x_missing))
        l_con = self.cl(x_incomp_con, x_missing_con)
        return l_con

    def encode_proj(self, x):
        x_emb = self.ae.encode(x)
        return x_emb

    def decode(self, x):
        x_emb, emb_loss = self.ae.quantize(x)
        x_rec = self.ae.decode(x_emb)
        return x_rec, emb_loss

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        z = self.ae.encode(x)
        return z

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        e, _ = self.ae.quantize(z)
        image = self.ae.decode(e)
        return image


