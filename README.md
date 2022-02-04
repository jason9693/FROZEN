# FROZEN

## Model 
```python
from frozen.models import FrozenModel

frozen = FrozenModel.from_pretrained("gpt2")
output = frozen(mok_img, mok_tokens)
```

## DATASET
Dataset preprocessing code is refered from [ViLT](https://github.com/dandelin/ViLT/tree/master/vilt/utils) Repository.
