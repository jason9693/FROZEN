from frozen.utils.write_vqa import make_arrow
make_arrow("./dataset/", "./dataset/arrows/")

from frozen.utils.write_coco_karpathy import make_arrow
make_arrow("./dataset/coco/", "./dataset/coco/arrows/")