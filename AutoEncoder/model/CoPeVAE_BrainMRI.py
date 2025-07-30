import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

# from AutoEncoder.VQVAE.vqvae import VQVAE
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
        self.classifier_number = similarity_head(args=self.args, in_dim=self.args.proj_dim, out_dim=3 if args.dataset == 'BraTS' else 2)
        self.classifier_positioning = similarity_head(args=self.args, in_dim=self.args.proj_dim, out_dim=self.args.modality_num)

        self.cl = ContrastiveLoss(self.args)


    def forward(self, x_incomp, x_missing, missing_number, missing_label):

        b, _, h, w, d = x_incomp.shape
        x_incomp = rearrange(x_incomp, 'b c h w d -> (b c) h w d').unsqueeze(1).contiguous()
        x_missing = rearrange(x_missing, 'b c h w d -> (b c) h w d').unsqueeze(1).contiguous()

        incomp_emb = self.encode_proj(x_incomp)
        missing_emb = self.encode_proj(x_missing)

        prompt_loc, l_loc = self.incompleteness_positioning(b, incomp_emb, missing_label)
        prompt_num, l_len = self.missing_number_detection(b, incomp_emb, missing_number)
        prompt_con, l_con = self.missing_modal_assessment(b, incomp_emb, missing_emb)

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
        b, c, h, w, d = x_in.shape
        x_in = rearrange(x_in, 'b c h w d -> (b c) h w d').unsqueeze(1).contiguous()
        z = self.ae.encode(x_in)
        x_emb, _ = self.ae.quantize(z)
        x_rec = self.ae.decode(x_emb)
        return x_in, x_rec


    def get_condition(self, x):
        b, _, _, _, _ = x.shape
        x_in = rearrange(x, 'b c h w d -> (b c) h w d').unsqueeze(1).contiguous()

        x_proj = self.encode_proj(x_in)

        len_proj = self.proj_head1(x_proj)
        loc_proj = self.proj_head2(x_proj)
        con_proj = self.proj_head3(x_proj)
        _, d = len_proj.shape
        len_proj = len_proj.view(b, -1, d).mean(dim=1)
        loc_proj = loc_proj.view(b, -1, d).mean(dim=1)
        con_proj = con_proj.view(b, -1, d).mean(dim=1)
        out_condition = torch.stack([len_proj, loc_proj, con_proj], dim=1)
        return out_condition

    def missing_number_detection(self, b, x_incomp, missing_number):
        x_incomp_out = self.proj_head1(x_incomp)
        _, d = x_incomp_out.shape
        x_incomp_out = x_incomp_out.view(b, -1, d).mean(dim=1)
        l_len = self.classifier_number(x_incomp_out, missing_number)
        return x_incomp_out, l_len

    def incompleteness_positioning(self, b, x_incomp, missing_idx):
        x_incomp_out = self.proj_head2(x_incomp)
        _, d = x_incomp_out.shape
        x_incomp_out = x_incomp_out.view(b, -1, d).mean(dim=1)
        l_loc_out = self.classifier_positioning(x_incomp_out, missing_idx)
        return x_incomp_out, l_loc_out

    def missing_modal_assessment(self, b, x_incomp, x_missing):
        x_incomp_out = self.proj_head3(x_incomp)
        x_missing_out = self.proj_head3(x_missing)
        _, d = x_incomp_out.shape
        x_incomp_out = x_incomp_out.view(b, -1, d).mean(dim=1)
        _, d = x_missing_out.shape
        x_missing_out = x_missing_out.view(b, -1, d).mean(dim=1)
        x_incomp_con = self.contrastive_head(x_incomp_out)
        x_missing_con = self.contrastive_head(x_missing_out)
        l_con = self.cl(x_incomp_con, x_missing_con)
        return x_incomp_out, l_con

    def encode_proj(self, x):
        x_emb = self.ae.encode(x)
        return x_emb

    def decode(self, x):
        x_emb, emb_loss = self.ae.quantize(x)
        x_rec = self.ae.decode(x_emb)
        return x_rec, emb_loss


    def encode_stage_2_inputs(self, x_in: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x_in.shape
        x = rearrange(x_in, 'b c h w d -> (b c) h w d').unsqueeze(1).contiguous()
        z = self.ae.encode(x)
        z = rearrange(z, '(b c) s h w d -> b c s h w d', b=b, c=c).contiguous()
        return z

    def decode_stage_2_outputs(self, z_in: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _, _ = z_in.shape
        z = rearrange(z_in, 'b c s h w d -> (b c) s h w d').contiguous()
        e, _ = self.ae.quantize(z)
        image = self.ae.decode(e)
        _, _, h, w, d = image.shape
        image = image.view(b, c, h, w, d)
        return image



