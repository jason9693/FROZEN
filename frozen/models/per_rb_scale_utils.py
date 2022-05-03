import warnings
from frozen.models.layers import PerResidualScaledEncoderBlock, PerResidualScaledDecoderBlock


class PerResidualDynamicLossScaler:
    def __init__(
        self,
        model,
        init_scale_factor,
        update_factor,
        backoff_factor
    ):
        self.model = model
        self.scale_factor = init_scale_factor
        self.update_factor = update_factor
        self.backoff_factor = backoff_factor

    def check_overflow(self):
        is_overflow = False
        for param in self.model.params:
            cpu_sum = float(param.grad.float().sum())
            if cpu_sum in (float('inf'), -float('inf')) or cpu_sum != cpu_sum:
                is_overflow = True
        return is_overflow

    def update_scale_factor(self):
        is_overflow = self.check_overflow()
        if is_overflow:
            self.scale_factor *= self.backoff_factor
        else:
            self.scale_factor *= self.update_factor
        for module in self.model.modules():
            if isinstance(module, PerResidualScaledEncoderBlock) or isinstance(module, PerResidualScaledDecoderBlock):
                module.update_scale_factor(self.scale_factor)
        return is_overflow

    def step(self, optimizer):
        if self.update_scale_factor():
            warnings.warn('Gradient overflow is occurred; SKIPPING update parameters at this step.')
        else:
            optimizer.step()



