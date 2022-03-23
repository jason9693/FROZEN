from frozen.utils.write_vqa import make_arrow
make_arrow("/nas/public/dataset/VQAv2", "/project/FROZEN/dataset/VQAv2/arrows")

from frozen.utils.write_coco_karpathy import make_arrow
make_arrow("/nas/public/dataset/coco2014", "/project/FROZEN/dataset/coco/arrows/")