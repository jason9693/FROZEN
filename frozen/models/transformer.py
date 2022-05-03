from torch import nn

from frozen.models.layers import PerResidualScaledEncoderBlock, PerResidualScaledDecoderBlock


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.expand_ratio = config.expand_ratio
        self.num_heads = config.num_heads
        self.num_layers = config.num_encoder_layers
        self.dropout = config.dropout
        self.attn_bias = config.attn_bias
        self.use_mixed_precision = config.precision == 16
        self.init_scale_factor = config.init_scale_factor
        layers = [
            PerResidualScaledEncoderBlock(
                self.embed_dim,
                self.num_heads,
                self.expand_ratio,
                self.dropout,
                self.attn_bias,
                self.use_mixed_precision,
                self.init_scale_factor
            )
            for _ in range(self.num_layers)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.expand_ratio = config.expand_ratio
        self.num_heads = config.num_heads
        self.num_layers = config.num_encoder_layers
        self.dropout = config.dropout
        self.attn_bias = config.attn_bias
        self.use_mixed_precision = config.precision == 16
        self.init_scale_factor = config.init_scale_factor
        layers = [
            PerResidualScaledDecoderBlock(
                self.embed_dim,
                self.num_heads,
                self.expand_ratio,
                self.dropout,
                self.attn_bias,
                self.use_mixed_precision,
                self.init_scale_factor
            )
            for _ in range(self.num_layers)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, key, value, attention_mask=None):
        output = x
        for layer in self.layers:
            output = layer(output, key, value, attention_mask)
        return output


class BaseTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, input_embeds, decode_embeds, attention_mask):
        encode_embeds = self.encoder(input_embeds)
        logits = self.decoder(decode_embeds, encode_embeds, encode_embeds, attention_mask)
        return dict(embeds=encode_embeds, logits=logits)


class Gen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BaseTransformer(config)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, input_embeds, decode_embeds, attention_mask):
        output = self.model(input_embeds, decode_embeds, attention_mask)
        output['pred'] = self.head(output['logits'])
        return output


