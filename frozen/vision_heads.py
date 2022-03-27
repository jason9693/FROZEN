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


class InteractiveAttention(nn.Module):
    def __init__(self, dim, num_vision_tokens, num_heads=8, qkv_bias=False, num_output_tokens=None):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_vision_tokens = num_vision_tokens
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.num_output_tokens = num_output_tokens
        head_dim = dim//num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        if num_output_tokens is not None:
            self.token_proj = nn.Linear(num_vision_tokens, num_output_tokens)
        else:
            self.token_proj = nn.Identity()

    def forward(self, x):
        output = x
        b, n, d = output.shape
        q = self.q(output[:, :self.num_vision_tokens])
        q = q.reshape(b, self.num_vision_tokens, 3, self.num_heads, d//self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(output).reshape(b, n, 3, self.num_heads, d//self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q@k.transpose(-2, -1))*self.scale
        attn = attn.softmax(dim=-1)

        output = (attn@v).transpose(1, 2).reshape(b, n, d)
        output = self.proj(output)
        output = self.token_proj(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = torch.cat([output, x[:, self.num_vision_tokens:]], dim=1)
        return output


class VisionAttentionHead(nn.Module):
    def __init__(
        self,
        dim,
        num_vision_tokens,
        num_attentions=2,
        num_heads=8,
        qkv_bias=False,
        num_output_tokens=None
    ):
        super().__init__()
        self.dim = dim
        self.num_vision_tokens = num_vision_tokens
        self.num_attentions = num_attentions
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.num_output_tokens = num_output_tokens
        attn = [
            InteractiveAttention(dim, num_vision_tokens, num_heads, qkv_bias, None)
            for _ in range(num_attentions-1)
        ]
        attn.append(InteractiveAttention(dim, num_vision_tokens, num_heads, qkv_bias, num_output_tokens))
        self.attn = nn.Sequential(*attn)

    def forward(self, v_tokens, nlp_tokens):
        tokens = torch.cat([v_tokens, nlp_tokens], dim=1)
        output = self.attn(tokens)
        output = output[:, :self.num_vision_tokens]
        return output


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
    head_module = head(feat_dim, embed_dim, num_global_tokens, num_local_tokens)
    setattr(model, attr_key, head_module)
    model.num_tokens = head_module.num_tokens
    if freeze_model:
        ENCODER_FACTORY[encoder_name]['freeze_func'](model, verbose=True)
