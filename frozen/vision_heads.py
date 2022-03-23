import torch
from einops import rearrange
from torch import nn


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
    def __init__(self, feat_dim, embed_dim, num_local_tokens):
        super().__init__()
        if isinstance(num_local_tokens, int):
            num_local_tokens = (num_local_tokens, int(num_local_tokens*1333./800.))
        self.pool = nn.AdaptiveAvgPool2d(num_local_tokens)
        self.proj = nn.Linear(feat_dim, embed_dim)
        self.num_tokens = num_local_tokens[0]*num_local_tokens[1]

    def forward(self, features):
        output = self.pool(features)
        output = rearrange(output, 'b d h w -> b (h w) d')
        output = self.proj(output)
        return output


class DuelEmbeddingHead(nn.Module):
    def __init__(self, feat_dim, embed_dim, num_global_tokens, num_local_tokens):
        super().__init__()
        self.global_head = GlobalEmbeddingHead(feat_dim, embed_dim, num_global_tokens)
        self.local_head = LocalEmbeddingHead(feat_dim, embed_dim, num_local_tokens)
        self.num_tokens = self.global_head.num_tokens+self.local_head.num_tokens

    def forward(self, features):
        return torch.cat([self.global_head(features), self.local_head(features)], dim=1)


ENCODER_FACTORY = dict(
    nf_resnet50=dict(
        attr_keys='head',
        feat_dim=2048
    ),
    resnet50=dict(
        attr_keys=['global_pool', 'fc'],
        feat_dim=2048
    )
)
HEAD_FACTORY = {
    "duel": lambda feat_dim, embed_dim, num_global_tokens, num_local_tokens:
    DuelEmbeddingHead(feat_dim, embed_dim, num_global_tokens, num_local_tokens),
    "global": lambda feat_dim, embed_dim, num_global_tokens, num_local_tokens:
    GlobalEmbeddingHead(feat_dim, embed_dim, num_global_tokens),
    "local": lambda feat_dim, embed_dim, num_global_tokens, num_local_tokens:
    LocalEmbeddingHead(feat_dim, embed_dim, num_local_tokens)
}


def wrap_vis_encoder(
    model,
    embed_dim,
    num_global_tokens,
    num_local_tokens,
    encoder_name,
    vis_mode
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
    head_module = head(feat_dim, embed_dim, num_global_tokens, num_local_tokens)
    setattr(model, attr_key, head_module)
    model.num_tokens = head_module.num_tokens


