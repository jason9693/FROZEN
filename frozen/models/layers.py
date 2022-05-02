import math

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


class ConvPatchEmbed(nn.Module):
    def __init__(self, dim=384, patch_size=16):
        super().__init__()
        in_dim = 3
        out_dim = 2*dim//patch_size
        to_patch_embed = []
        num_ds = int(math.log2(patch_size))
        for i in range(num_ds-1):
            to_patch_embed.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 3, 2, 1),
                    nn.BatchNorm2d(out_dim),
                    nn.GELU()
                )
            )
            in_dim = out_dim
            out_dim *= 2
        to_patch_embed.append(nn.Conv2d(in_dim, dim, 3, 2, 1))
        to_patch_embed.append(
            Rearrange('b d h w -> b (h w) d')
        )
        self.to_patch_embed = nn.Sequential(*to_patch_embed)

    def forward(self, x):
        return self.forward_features(x)

    def forward_features(self, x):
        return self.to_patch_embed(x)


class PatchEmbedToImage(nn.Module):
    def __init__(self, dim=384, patch_size=16):
        super().__init__()
        in_dim = dim
        out_dim = dim//2
        to_patch_embed = []
        num_us = int(math.log2(patch_size))
        for i in range(num_us-1):
            to_patch_embed.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_dim, out_dim, 3, 2, 1),
                    # nn.BatchNorm2d(out_dim),
                    nn.GroupNorm(num_groups=32, num_channels=out_dim),
                    nn.GELU()
                )
            )
            in_dim = out_dim
            out_dim //= 2
        to_patch_embed.append(nn.ConvTranspose2d(in_dim, 3, 3, 2, 1))
        self.upsample = nn.Sequential(*to_patch_embed)

    def forward(self, x):
        return self.forward_features(x)

    def forward_features(self, x):
        x = self.upsample(x)
        return x


def conv3_bn_gelu(in_dim, out_dim, stride):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 3, stride, 1),
        nn.BatchNorm2d(out_dim),
        nn.GELU()
    )


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, is_decoder=False):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim//num_heads
        self.is_decoder = is_decoder
        if is_decoder:
            self.qkv = nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=bias),
                Rearrange('b n (h d) -> b h n d', h=self.num_heads)
            )
        else:
            self.qkv = nn.Sequential(
                nn.Linear(embed_dim, embed_dim*3, bias=bias),
                Rearrange('b n (h d) -> b h n d', h=self.num_heads)
            )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x,
        key=None,
        value=None,
        attention_mask=None):
        # TODO: This implementation does not consider model-parallelism yet, so it might require modifying after
        if self.is_decoder:
            assert key is not None and value is not None, 'key and value have to exist for decoder'
            query = self.qkv(x)
        else:
            query, key, value = torch.chunk(self.qkv(x), chunks=3, dim=-1)
        attention_weight = torch.einsum('b h i d, b h j d -> b h i j', query, key)
        if attention_mask is not None:
            attention_weight = attention_weight+attention_mask
        attention_weight = torch.softmax(attention_weight, dim=-1)
        output = torch.einsum('b h i j, b h j d -> b h i d', attention_weight, value)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.proj(output)
        return output


class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_dim, expand_ratio=4.):
        super().__init__()
        self.in_dim = embed_dim
        self.h_dim = int(embed_dim*expand_ratio)
        self.out_dim = embed_dim
        self.fc0 = nn.Linear(self.in_dim, self.h_dim)
        self.fc1 = nn.Linear(self.h_dim, self.out_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        output = self.fc0(x)
        output = self.gelu(output)
        output = self.fc1(output)
        return output


class PrecisionScaleFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale_factor):
        ctx.save_for_backward(scale_factor)
        return x

    @staticmethod
    def backward(ctx, grad):
        scale_factor = ctx.saved_tensors[0]
        scaled_grad = grad*scale_factor
        filtered_grad = scaled_grad*torch.isinf(scaled_grad).float()*torch.isnan(scaled_grad).float()
        return filtered_grad


class ToFloat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.float()

    @staticmethod
    def backward(ctx, grad):
        return grad.half()


class ToHalf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.half()

    @staticmethod
    def backward(ctx, grad):
        return grad.float()


def scale_filter_wrapper(x, norm, fx, scale_factor):
    inv_scale_factor = scale_factor.double().reciprocal().float()
    output = PrecisionScaleFilter.apply(x, inv_scale_factor)  # unscale and filter
    output = norm(output)
    output = ToHalf.apply(output)
    output = fx(output)
    output = ToFloat.apply(output)
    output = PrecisionScaleFilter.apply(output, scale_factor)  # scale and filter
    return output


class PerResidualFP16ScaledTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        expand_ratio=4.,
        init_scale_factor=math.pow(2., 13.),
        update_factor=math.pow(2., 1./1000.),
        backoff_factor=math.sqrt(0.5),
        dropout=0.,
        bias=True,
        is_decoder=False
    ):
        super().__init__()
        self.scale_factor = init_scale_factor
        self.update_factor = update_factor
        self.backoff_factor = backoff_factor
        self.embed_dim = embed_dim
        self.expand_ratio = expand_ratio
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout, bias, is_decoder)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = MultiLayerPerceptron(embed_dim, expand_ratio)

    @torch.cuda.amp.autocast(False)
    def forward(self, x, key=None, value=None, attention_mask=None):
        scale_factor = torch.tensor(self.scale_factor, dtype=torch.float32).to(x.device)
        res = output = x
        attn_fx = lambda _x: self.attn(_x, key, value, attention_mask)
        output = scale_filter_wrapper(output, self.attn_norm, attn_fx, scale_factor)
        res = output = output+res
        output = scale_filter_wrapper(output, self.mlp_norm, self.mlp, scale_factor)
        output = output+res
        return output

    def update_scale_factor(self, is_overflow):
        mul_factor = self.backoff_factor if is_overflow else self.update_factor
        self.scale_factor *= mul_factor


