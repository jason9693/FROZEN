import warnings
from contextlib import contextmanager

from frozen.models.layers import PerResidualFP16ScaledTransformerBlock


def check_overflow(params):
    for param in params:
        cpu_sum = float(param.grad.float().sum())
        if cpu_sum in (float('inf'), -float('inf')) or cpu_sum != cpu_sum:
            return True
    return False


def update_scale_factor(model):
    is_overflow = check_overflow(model.parameters())
    for module in model.modules():
        if isinstance(module, PerResidualFP16ScaledTransformerBlock):
            module.update_scale_factor(is_overflow)
    return is_overflow


# for training w/o pytorch-lightning
# for pytorch-lightning, you have to implement optimizer_step manually
@contextmanager
def fp16_step(model, optimizer, verbose=True):
    yield
    is_overflow = update_scale_factor(model)
    if not is_overflow:
        optimizer.step()
    elif verbose:
        warnings.warn('Gradient overflow is detected; SKIPPING gradient update.')

