# --------------------------------------------------------
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# Based on https://github.com/lucidrains/rotary-embedding-torch
# --------------------------------------------------------'

from math import pi

import torch
from torch import nn
from einops import rearrange, repeat



def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbedding_3D(nn.Module):
    def __init__(
        self,
        dim,
        depth=4,
        height=12,
        width=12,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        self.depth = depth
        self.height = height
        self.width = width

        dim_d = dim // 4
        dim_h = dim // 8 * 3
        dim_w = dim // 8 * 3

        freqs_d = 1. / (theta ** (torch.arange(0, dim_d, 2)[:(dim_d // 2)].float() / dim_d))
        freqs_h = 1. / (theta ** (torch.arange(0, dim_h, 2)[:(dim_h // 2)].float() / dim_h))
        freqs_w = 1. / (theta ** (torch.arange(0, dim_w, 2)[:(dim_w // 2)].float() / dim_w))

        grid_d = torch.arange(depth, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w = torch.arange(width, dtype=torch.float32)

        freqs_d = torch.einsum("..., f -> ... f", grid_d, freqs_d)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)

        freqs_d = repeat(freqs_d, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat(
            (freqs_d[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]),
            dim=-1,
        )
        freqs = freqs.contiguous()

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, t):
        def reshape_freq(freqs):
            freqs = freqs[: self.depth, : self.height, : self.width].contiguous()
            freqs = rearrange(freqs, "t h w d -> (t h w) d")
            freqs = freqs.unsqueeze(0).unsqueeze(0)
            return freqs

        freqs_cos = reshape_freq(self.freqs_cos).to(t.dtype)
        freqs_sin = reshape_freq(self.freqs_sin).to(t.dtype)
        return t * freqs_cos + rotate_half(t) * freqs_sin


class VisionRotaryEmbedding_2D(nn.Module):
    def __init__(
        self,
        dim,
        height=12,
        width=12,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        self.height = height
        self.width = width

        dim_h = dim // 2
        dim_w = dim // 2

        freqs_h = 1. / (theta ** (torch.arange(0, dim_h, 2)[:(dim_h // 2)].float() / dim_h))
        freqs_w = 1. / (theta ** (torch.arange(0, dim_w, 2)[:(dim_w // 2)].float() / dim_w))

        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w = torch.arange(width, dtype=torch.float32)

        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)

        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)
        freqs = freqs.contiguous()

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, t):
        def reshape_freq(freqs):
            freqs = freqs[: self.height, : self.width].contiguous()
            freqs = rearrange(freqs, "h w d -> (h w) d")
            freqs = freqs.unsqueeze(0).unsqueeze(0)
            return freqs

        freqs_cos = reshape_freq(self.freqs_cos).to(t.dtype)
        freqs_sin = reshape_freq(self.freqs_sin).to(t.dtype)
        return t * freqs_cos + rotate_half(t) * freqs_sin

class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t): return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin