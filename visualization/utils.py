import json
import os.path
import uuid
from datetime import datetime
from io import BytesIO

from PIL import Image
import numpy as np

from frozen.transforms import pixelbert_transform


def convert_to_inputs(img_input, image_size, transform=pixelbert_transform):
    if isinstance(img_input, BytesIO):
        image = Image.open(img_input).convert('RGB')
    else:
        image = img_input
    image_tensor = transform(size=image_size)(image).unsqueeze(0)
    return image, image_tensor


def draw_transparent_heatmap(image, value):
    value = (value-value.min())/(value.max()-value.min())
    value = value.data.cpu().numpy()
    _w, _h = image.size
    overlay = Image.fromarray(np.uint8(value*255), "L").resize((_w, _h), resample=Image.NEAREST)
    image_rgba = image.copy()
    image_rgba.putalpha(overlay)
    return image_rgba


class ImageBasedLocalData:
    def __init__(self, name, root_dir):
        self.name = name
        self.root_dir = root_dir
        self.state_file_path = os.path.join(self.root_dir, f'{name}.json')
        self._load()
        self.image_dir = os.path.join(self.root_dir, 'files')
        os.makedirs(self.image_dir, exist_ok=True)

    def add(self, image, state_dict, key=None):
        image_path = os.path.join(self.image_dir, f'{uuid.uuid4()}')
        image.save(image_path)
        state_dict['image_path'] = image_path
        key = key or datetime.now().strftime('%Y%d%m%H%M%S')
        self.state_dict[key] = state_dict

    def save(self):
        with open(self.state_file_path, 'w') as f:
            json.dump(self.state_dict, f)

    def add_and_save(self, image, state_dict, key=None):
        self.add(image, state_dict, key)
        self.save()

    def _load(self):
        if os.path.exists(self.state_file_path):
            with open(self.state_file_path, 'r') as f:
                self.state_dict = json.load(f)
        else:
            self.state_dict = dict()

    def keys(self):
        return self.state_dict.keys()

    def values(self):
        return self.state_dict.values()

    def items(self):
        return self.state_dict.items()

    def __iter__(self):
        for key in self.state_dict:
            yield key
