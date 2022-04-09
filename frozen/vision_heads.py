import math

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim//num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, d//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1))*self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, d)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim//num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, vis_embed, nlp_embed):
        embed = torch.cat([vis_embed, nlp_embed], dim=1)
        b, n, d = embed.shape
        q = self.q(nlp_embed).reshape(b, nlp_embed.size(1), 1, self.num_heads, d//self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(embed).reshape(b, n, 2, self.num_heads, d//self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1))*self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        embed = (attn @ v).transpose(1, 2).reshape(b, nlp_embed.size(1), d)
        embed = self.proj(embed)
        embed = self.proj_drop(embed)
        return embed


class GlobalEmbeddingHead(nn.Module):
    def __init__(self, feat_dim, embed_dim, num_global_tokens):
        super().__init__()
        self.proj = nn.Linear(feat_dim, num_global_tokens*embed_dim)
        self.num_tokens = num_global_tokens

    def forward(self, features):
        output = self.proj(features.mean(dim=[2, 3]))
        output = rearrange(output, 'b (g d) -> b g d', g=self.num_tokens)
        return output


class LocalEmbeddingHead(nn.Module):
    def __init__(self, feat_dim, embed_dim, local_output_size):
        super().__init__()
        if isinstance(local_output_size, int):
            local_output_size = (local_output_size, int(local_output_size*1333./800.))
        self.pool = nn.AdaptiveAvgPool2d(local_output_size)
        self.proj = nn.Linear(feat_dim, embed_dim)
        self.num_tokens = local_output_size[0]*local_output_size[1]

    def forward(self, features):
        output = self.pool(features)
        output = rearrange(output, 'b d h w -> b (h w) d')
        output = self.proj(output)
        return output


class DuelEmbeddingHead(nn.Module):
    def __init__(self, feat_dim, embed_dim, num_global_tokens, local_output_size):
        super().__init__()
        self.global_head = GlobalEmbeddingHead(feat_dim, embed_dim, num_global_tokens)
        self.local_head = LocalEmbeddingHead(feat_dim, embed_dim, local_output_size)
        self.num_tokens = self.global_head.num_tokens+self.local_head.num_tokens

    def forward(self, features):
        return torch.cat([self.global_head(features), self.local_head(features)], dim=1)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.gelu = nn.GELU()
        self.fc0 = nn.Linear(in_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        output = self.fc0(x)
        output = self.gelu(output)
        output = self.fc1(output)
        return output


class InteractionModule(nn.Module):
    def __init__(
        self,
        dim,
        num_input_tokens,
        num_vis_tokens=None,
        num_heads=8,
        qkv_bias=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_input_tokens = num_input_tokens
        self.num_vis_tokens = num_vis_tokens or num_input_tokens
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.norm0 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = MultiLayerPerceptron(dim, 4*dim, dim)
        if num_vis_tokens != num_input_tokens:
            self.token_proj = nn.Linear(num_input_tokens, num_vis_tokens)
        else:
            self.token_proj = nn.Identity()

    def forward(self, vis_embed):
        res = output = vis_embed
        output = self.norm0(output)
        res = output = res+self.attn(output)
        output = self.norm1(output)
        output = res+self.mlp(output)
        output = self.token_proj(output.permute(0, 2, 1)).permute(0, 2, 1)
        return output


class DeepFusion(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.norm0 = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = MultiLayerPerceptron(dim, 4*dim, dim)

    def forward(self, vis_embed, nlp_embed):
        res = output = torch.cat([vis_embed, nlp_embed], dim=1)
        output = self.norm0(output)
        idx = vis_embed.size(1)
        res = output = res[:, idx:]+self.attn(output[:idx], output[:, idx:])
        output = self.norm1(output)
        output = res+self.mlp(output)
        return output


class ConcatHead(nn.Module):
    def __init__(self, num_input_tokens, num_vis_tokens=None):
        super().__init__()
        self.num_input_tokens = num_input_tokens
        self.num_vis_tokens = num_vis_tokens or num_input_tokens

    def forward(self, vis_embed, nlp_embed, attn_mask):
        output = torch.cat([vis_embed, nlp_embed], dim=1)
        device = output.device
        attn_mask = torch.cat([torch.ones(*vis_embed.size()[:2]).to(device), attn_mask], 1)
        return output, attn_mask

    def get_vis_label(self, img):
        return -100*torch.ones(img.size(0), self.num_vis_tokens).to(img.device)


class VisionAttentionHead(nn.Module):
    def __init__(
        self,
        dim,
        num_input_tokens,
        num_vis_tokens=None,
        num_attentions=2,
        num_heads=8,
        qkv_bias=False
    ):
        super().__init__()
        self.dim = dim
        self.num_input_tokens = num_input_tokens
        self.num_vis_tokens = num_vis_tokens or num_input_tokens
        self.num_attentions = num_attentions
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn = nn.ModuleList([
            InteractionModule(
                dim,
                num_heads,
                qkv_bias,
                num_input_tokens,
                num_vis_tokens if i == num_attentions-1 else num_input_tokens
            )
            for i in range(num_attentions)
        ])

    def forward(self, vis_embed, nlp_embed, attn_mask):
        output = vis_embed
        for attn in self.attn:
            output = attn(output)
        device = output.device
        attn_mask = torch.cat([torch.ones(*output.size()[:2]).to(device), attn_mask], 1)
        return torch.cat([output, nlp_embed], dim=1), attn_mask

    def get_vis_label(self, img):
        return -100*torch.ones(img.size(0), self.num_vis_tokens).to(img.device)


class DeepFusionHead(nn.Module):
    def __init__(
        self,
        dim,
        num_input_tokens,
        num_vis_tokens=None,
        num_attentions=2,
        num_heads=8,
        qkv_bias=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_input_tokens = num_input_tokens
        self.num_vis_tokens = num_vis_tokens or num_input_tokens
        self.num_attentions = num_attentions
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        attn = []
        for i in range(num_attentions):
            if i == num_attentions-1:
                attn_module = DeepFusion(dim, num_heads, qkv_bias)
            else:
                attn_module = InteractionModule(dim, None, None, num_heads, qkv_bias)
            attn.append(attn_module)
        self.attn = nn.ModuleList(attn)

    def forward(self, vis_embed, nlp_embed, attn_mask):
        output = vis_embed
        for attn in self.attn[:-1]:
            output = attn(output)
        output = self.attn[-1](output, nlp_embed)
        return output, attn_mask

    def get_vis_label(self, img):
        return


def freeze(module, verbose=False):
    for name, p in module.named_parameters():
        p.requires_grad = False
        if verbose:
            print(f'* Parameter {name} has been frozen.')


def _freeze_nfnet(model, verbose=False):
    freeze(model.stem, verbose)
    for stage in model.stages[:-1]:
        freeze(stage, verbose)


ENCODER_FACTORY = dict(
    nf_resnet50=dict(
        attr_keys='head',
        freeze_func=_freeze_nfnet,
        feat_dim=2048
    ),
    resnet50=dict(
        attr_keys=['global_pool', 'fc'],
        freeze_func=None,
        feat_dim=2048
    ),
    linear=dict(
        attr_keys='head',
        freeze_func=None,
        feat_dim=384
    ),
    conv=dict(
        attr_keys='head',
        freeze_func=None,
        feat_dim=384
    )
)
HEAD_FACTORY = {
    "duel": lambda feat_dim, embed_dim, num_global_tokens, local_output_size:
    DuelEmbeddingHead(feat_dim, embed_dim, num_global_tokens, local_output_size),
    "global": lambda feat_dim, embed_dim, num_global_tokens, local_output_size:
    GlobalEmbeddingHead(feat_dim, embed_dim, num_global_tokens),
    "local": lambda feat_dim, embed_dim, num_global_tokens, local_output_size:
    LocalEmbeddingHead(feat_dim, embed_dim, local_output_size)
}
VISION_HEAD_FACTORY = {
    'interactive': lambda dim, num_input_tokens, num_vis_tokens=None, num_attentions=2, num_heads=8, qkv_bias=True:
    VisionAttentionHead(dim, num_input_tokens, num_vis_tokens, num_attentions, num_heads, qkv_bias),
    'deep-fusion': lambda dim, num_input_tokens, num_vis_tokens=None, num_attentions=2, num_heads=8, qkv_bias=True:
    DeepFusionHead(dim, num_input_tokens, num_attentions, num_heads, qkv_bias),
    'concat': lambda dim, num_input_tokens, num_vis_tokens=None, num_attentions=1, num_heads=8, qkv_bias=True:
    ConcatHead(num_input_tokens)
}


def wrap_vis_encoder(
    model,
    embed_dim,
    num_global_tokens,
    local_output_size,
    encoder_name,
    vis_mode,
    freeze_model
):
    encoder_info = ENCODER_FACTORY[encoder_name]
    attr_keys = encoder_info['attr_keys']
    if isinstance(attr_keys, str):
        attr_keys = [attr_keys]
    feat_dim = encoder_info['feat_dim']
    head = HEAD_FACTORY[vis_mode]
    attr_key = attr_keys.pop(0)
    for id_key in attr_keys:
        setattr(model, id_key, nn.Identity())
    head_module = head(feat_dim, embed_dim, num_global_tokens, local_output_size)
    setattr(model, attr_key, head_module)
    model.num_tokens = head_module.num_tokens
    if freeze_model:
        ENCODER_FACTORY[encoder_name]['freeze_func'](model, verbose=False)


class LinearPatchEmbed(nn.Module):
    def __init__(self, dim=384, patch_size=16):
        super().__init__()
        self.to_patch_embed = nn.Sequential(
            Rearrange('b d (h p1) (w p2) -> b h w (p1 p2 d)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size*patch_size*3, dim),
            Rearrange('b h w d -> b d h w'),
        )
        self.head = nn.Identity()

    def forward(self, x):
        return self.head(self.to_patch_embed(x))


class Stem(nn.Module):
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
        self.to_patch_embed = nn.Sequential(*to_patch_embed)
        self.head = nn.Identity()

    def forward(self, x):
        return self.head(self.to_patch_embed(x))


