# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed
from LDM.model.rmsnorm import RMSNorm
from LDM.model.swiglu_ffn import SwiGLUFFN
from LDM.model.pos_embed import VisionRotaryEmbedding_3D, VisionRotaryEmbedding_2D


# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            fused_attn: bool = True,
            use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        if use_rmsnorm:
            norm_layer = RMSNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class PatchEmbed_3D(nn.Module):
    """ Voxel to Patch Embedding
    """
    def __init__(self, voxel_size=(24, 24, 8), patch_size=4, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (voxel_size[0] // patch_size[0]) * (voxel_size[1] // patch_size[1]) * (voxel_size[2] // patch_size[2])
        self.patch_xyz = (voxel_size[0] // patch_size[0], voxel_size[1] // patch_size[1], voxel_size[2] // patch_size[2])
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        x = x.float()
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core MDiT3D Model                                #
#################################################################################

class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=True,
        use_swiglu=True,
        use_rmsnorm=True,
        **block_kwargs
    ):
        super().__init__()
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size, eps=1e-4)
            self.norm2 = RMSNorm(hidden_size, eps=1e-4)

        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=True):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size, eps=1e-4)

        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MDiT3D(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=24,
        input_depths=4,
        patch_size=2,
        in_channels=8,
        hidden_size=768,
        prompt_size=512,
        prompt_num=3,
        condition_size=32,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size
        self.input_depths = input_depths
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm

        self.x_embedder = PatchEmbed_3D((input_size, input_size, input_depths), patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.prompt_embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(prompt_num * prompt_size, hidden_size, bias=True)
        )

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.spatial_embed = nn.Parameter(torch.zeros(1, input_depths // patch_size, hidden_size), requires_grad=False)
        self.hidden_size =  hidden_size

        if self.use_rope:
            head_dim = hidden_size // num_heads
            self.feat_rope = VisionRotaryEmbedding_3D(
                dim=head_dim,
                depth=input_depths// patch_size,
                height=input_size// patch_size,
                width=input_size// patch_size
            )

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
            use_qknorm = use_qknorm, use_swiglu = use_swiglu, use_rmsnorm = use_rmsnorm) for _ in range(depth)])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.input_size//self.patch_size), int(self.input_size//self.patch_size),
                                            int(self.input_depths//self.patch_size))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        spatial_embed = get_1d_sincos_spatial_embed(self.spatial_embed.shape[-1], self.spatial_embed.shape[-2])
        self.spatial_embed.data.copy_(torch.from_numpy(spatial_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in MDiT3D blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify_3D(self, x0):
        """
        input: (N, T, patch_size * patch_size * patch_size * C)    (N, 64, 8*8*8*3)
        voxels: (N, C, X, Y, Z)          (N, 3, 32, 32, 32)
        """
        c = self.out_channels
        p = self.patch_size
        x = y = self.input_size // self.patch_size
        z = self.input_depths // self.patch_size
        assert x * y * z == x0.shape[1]

        x0 = x0.reshape(shape=(x0.shape[0], x, y, z, p, p, p, c))
        x0 = torch.einsum('nxyzpqrc->ncxpyqzr', x0)
        points = x0.reshape(shape=(x0.shape[0], c, x * p, y * p, z * p))
        return points

    def forward(self, 
                x, 
                timesteps,
                y=None, 
                context=None,
                use_fp16=False):
        """
        Forward pass of MDiT3D.
        x: (N, D, C, H, W) tensor of volume inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if use_fp16:
            x = x.to(dtype=torch.float16)

        n_t = self.input_depths // self.patch_size

        batches, depths, channels, high, weight = x.shape
        x = rearrange(x, 'b d c h w -> b c h w d')
        x = self.x_embedder(x) + self.pos_embed  
        t = self.t_embedder(timesteps, use_fp16=use_fp16)
        timestep_planar = t
        timestep_spatial = repeat(t, 'n d -> (n c) d', c=self.x_embedder.num_patches // n_t)

        if context is not None:
            if context.shape[0] != batches:
                if context.shape[0] == 1:
                    context = context.repeat(batches, 1, 1)
        context = self.prompt_embedding_projection(context.reshape(batches, -1))
        prompt_embedding_spatial = repeat(context, 'n d -> (n c) d', c=self.x_embedder.num_patches // n_t)

        for i in range(0, len(self.blocks), 2):
            planar_block, spatial_block = self.blocks[i:i+2]
            # planar_block
            c = timestep_planar
            x  = planar_block(x, c, self.feat_rope)
            x = rearrange(x, 'b (n t) d -> (b n) t d', t=n_t)
            # spatial_block
            if i == 0:
                x = x + self.spatial_embed
            c = timestep_spatial + prompt_embedding_spatial
            x = spatial_block(x, c)
            x = rearrange(x, '(b n) t d -> b (n t) d', b=batches)

        c = timestep_planar
        x = self.final_layer(x, c)               
        x = self.unpatchify_3D(x)
        x = rearrange(x, 'b c h w d -> b d c h w')
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_3d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, grid_size_z, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # print('grid_size:', grid_size)
    grid_x = np.arange(grid_size_h, dtype=np.float32)
    grid_y = np.arange(grid_size_w, dtype=np.float32)
    grid_z = np.arange(grid_size_z, dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # here y goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size_h, grid_size_w, grid_size_z])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0
    # use half of dimensions to encode grid_h
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (X*Y*Z, D/3)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (X*Y*Z, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1) # (X*Y*Z, D)
    return emb


def get_1d_sincos_spatial_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb


#################################################################################
#                                   MDiT3D Configs                                  #
#################################################################################

def MDiT3D_B_1(**kwargs):
    return MDiT3D(depth=12, hidden_size=576, patch_size=1, num_heads=12, **kwargs)

def MDiT3D_B_2(**kwargs):
    return MDiT3D(depth=12, hidden_size=576, patch_size=2, num_heads=12, **kwargs)

def MDiT3D_B_4(**kwargs):
    return MDiT3D(depth=12, hidden_size=576, patch_size=4, num_heads=12, **kwargs)

def MDiT3D_B_8(**kwargs):
    return MDiT3D(depth=12, hidden_size=576, patch_size=8, num_heads=12, **kwargs)
    
def MDiT3D_S_1(**kwargs):
    return MDiT3D(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)
    
def MDiT3D_S_2(**kwargs):
    return MDiT3D(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def MDiT3D_S_4(**kwargs):
    return MDiT3D(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def MDiT3D_S_8(**kwargs):
    return MDiT3D(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)
    
def MDiT3D_T_1(**kwargs):
    return MDiT3D(depth=10, hidden_size=288, patch_size=1, num_heads=6, **kwargs)
    
def MDiT3D_T_2(**kwargs):
    return MDiT3D(depth=10, hidden_size=288, patch_size=2, num_heads=6, **kwargs)

def MDiT3D_T_4(**kwargs):
    return MDiT3D(depth=10, hidden_size=288, patch_size=4, num_heads=6, **kwargs)

def MDiT3D_T_8(**kwargs):
    return MDiT3D(depth=10, hidden_size=288, patch_size=8, num_heads=6, **kwargs)


MDiT3D_models = {
    'MDiT3D-B/1':  MDiT3D_B_1,   'MDiT3D-B/2':  MDiT3D_B_2,   'MDiT3D-B/4':  MDiT3D_B_4,   'MDiT3D-B/8':  MDiT3D_B_8,
    'MDiT3D-S/1':  MDiT3D_S_1,   'MDiT3D-S/2':  MDiT3D_S_2,   'MDiT3D-S/4':  MDiT3D_S_4,   'MDiT3D-S/8':  MDiT3D_S_8,
    'MDiT3D-T/1':  MDiT3D_T_1,   'MDiT3D-T/2':  MDiT3D_T_2,   'MDiT3D-T/4':  MDiT3D_T_4,   'MDiT3D-T/8':  MDiT3D_T_8,
}
