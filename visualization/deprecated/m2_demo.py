import glob
import json
import os
import uuid
from io import BytesIO

import numpy as np
import streamlit as st
import torch
from PIL import Image

from frozen.datamodules.datamodule_base import get_pretrained_tokenizer
from frozen.models.bifrost_translator import BiFrostTranslator
from frozen.transforms import pixelbert_transform

PYTHON_PATH = os.path.abspath('../..')


def get_all_checkpoint_paths(root_dir=os.path.join(PYTHON_PATH, 'M2'), path_format='M2*.ckpt'):
    paths = list(glob.glob(os.path.join(root_dir, path_format)))
    paths.sort()
    return paths


def convert_to_inputs(img_input, image_size):
    if isinstance(img_input, BytesIO):
        image = Image.open(img_input).convert('RGB')
    else:
        image = img_input
    image_tensor = pixelbert_transform(size=image_size)(image).unsqueeze(0)
    return image, image_tensor


def load_model(path):
    checkpoint = torch.load(path)
    model = BiFrostTranslator()
    model.load_state_dict(checkpoint['state_dict'])
    return model


@torch.no_grad()
def infer_and_export(
    model,
    image_tensor,
    max_length,
    ignore_eos,
):
    output = model.infer(image_tensor.cuda(), max_length, ignore_eos)
    decoded_text = model.tokenizer.decode(output)
    st.text(f'Translated Result: {output}')


def main():
    st.sidebar.title('M2 Demo')
    image_size = 384
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.cuda.set_device(torch.cuda.device_count()-1)  # use last device
    state = st.session_state
    checkpoint_list = get_all_checkpoint_paths()
    load_path = st.sidebar.selectbox(
        'Choose a checkpoint to load.',
        options=checkpoint_list,
        format_func=lambda path: path.split('/')[-1]
    )
    if not os.path.exists(load_path):
        st.sidebar.error(f'Cannot load the model: {load_path}')
    else:
        with st.spinner('Loading the selected model, please wait...'):
            if load_path in state:
                model = state[load_path]
            else:
                model = load_model(load_path)
                model.setup('test')
                model.eval().cuda()
                state[load_path] = model
        reload_model = st.sidebar.button('Reload Model')
        if reload_model:
            del state[load_path]
            st.experimental_rerun()
        max_length = st.sidebar.select_slider('Set the maximum length of output sentence.', list(range(10, 31)))
        ignore_eos = st.sidebar.checkbox('Ignore EOS')
        example_dir = os.path.join(os.getcwd(), 'examples')
        os.makedirs(example_dir, exist_ok=True)
        examples_path = os.path.join(example_dir, 'examples.json')
        if not os.path.exists(examples_path):
            with open(examples_path, 'w') as f:
                json.dump([], f)
        submit_modes = ['Upload your own image', 'Sample from examples']
        submit_mode = st.sidebar.selectbox('', submit_modes)
        if submit_mode == submit_modes[0]:
            image = image_tensor = None
            file = st.sidebar.file_uploader('Upload an image.')
            does_image_exist = False
            if file is None:
                if state.get('inputs') is not None:
                    image, image_tensor = state.inputs
                    does_image_exist = True
            else:
                image, image_tensor = convert_to_inputs(file, image_size)
                state.inputs = (image, image_tensor)
                does_image_exist = True
            if does_image_exist:
                st.image(image, caption='Input Image')
                infer_and_export(
                    model,
                    image_tensor,
                    max_length,
                    ignore_eos
                )
                add_to_examples = st.button('Add to Examples')
                if add_to_examples:
                    with open(examples_path, 'r') as f:
                        examples = json.load(f)
                    image_dir = os.path.join(example_dir, 'images')
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f'{uuid.uuid4()}.png')
                    new_example = dict()
                    new_example['image'] = image_path
                    examples.append(new_example)
                    image.save(image_path)
                    with open(examples_path, 'w') as f:
                        json.dump(examples, f)
                    st.success("Your example is added!")
                resubmit = st.sidebar.button('Resubmit')
                if resubmit:
                    st.experimental_rerun()
        elif submit_mode == submit_modes[1]:
            with open(examples_path, 'r') as f:
                examples = json.load(f)
            options = [f"{i}. {example['input_text']}" for i, example in enumerate(examples)]
            example = st.sidebar.selectbox('Choose an example set.', options)
            idx = options.index(example)
            image_path = examples[idx]['image']
            image = Image.open(image_path).convert('RGB')
            st.image(image, caption='Input Image')
            _, image_tensor = convert_to_inputs(image, image_size)
            infer_and_export(
                model,
                image_tensor,
                max_length,
                ignore_eos
            )
            resubmit = st.sidebar.button('Resubmit')
            if resubmit:
                st.experimental_rerun()
            delete = st.sidebar.button('Delete this example')
            if delete:
                os.remove(image_path)
                del examples[idx]
                st.sidebar.error("Example is deleted.")
                with open(examples_path, 'w') as f:
                    json.dump(examples, f)
                st.experimental_rerun()


if __name__ == '__main__':
    main()
