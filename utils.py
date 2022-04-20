def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def defrost_module(module):
    for p in module.parameters():
        p.requires_grad = True

