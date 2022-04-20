import math
from einops import rearrange

from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


class MinMaxResize:
    def __init__(self, shorter=800, longer=1333):
        self.min = shorter
        self.max = longer

    def __call__(self, x):
        w, h = x.size
        scale = self.min / min(w, h)
        if h < w:
            newh, neww = self.min, scale * w
        else:
            newh, neww = scale * h, self.min

        if max(newh, neww) > self.max:
            scale = self.max / max(newh, neww)
            newh = newh * scale
            neww = neww * scale

        newh, neww = int(newh + 0.5), int(neww + 0.5)
        newh, neww = newh // 32 * 32, neww // 32 * 32

        return x.resize((neww, newh), resample=Image.BICUBIC)


class CorruptImage:
    def __init__(self, corrupt_prob, patch_size):
        self.patch_size = patch_size
        self.corrupt_prob = corrupt_prob

    def __call__(self, x1, x2):
        x1 = self._pad_image(x1)
        x2 = self._cut_image(x2)
        x1 = rearrange(x1, 'b d (h p1) (w p2) -> b d h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        x2 = rearrange(x2, 'b d (h p1) (w p2) -> b d h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        _, _, h1, w1 = x1.size()
        _, _, h2, w2 = x2.size()
        overlap_region = [min(h1, h2), min(w1, w2)]

    def _cut_image(self, x):
        b, d, h, w = x.size()
        h_cut_idx = h % self.patch_size
        w_cut_idx = w % self.patch_size
        x = x[:, :, :h_cut_idx, :w_cut_idx]
        return x

    def _pad_image(self, x):
        b, d, h, w = x.size()
        h_pad_size = self.patch_size*math.ceil(h/self.patch_size)-h
        w_pad_size = self.patch_size*math.ceil(w/self.patch_size)-w
        h_pad_size = [h_pad_size//2, h_pad_size-h_pad_size//2]
        w_pad_size = [w_pad_size//2, w_pad_size-w_pad_size//2]
        return F.pad(x, w_pad_size+h_pad_size)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


# This is simple maximum entropy normalization performed in Inception paper
inception_normalize = transforms.Compose(
    [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)

# ViT uses simple non-biased inception normalization
# https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py#L132
inception_unnormalize = transforms.Compose(
    [UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)
