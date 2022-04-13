def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


