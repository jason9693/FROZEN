import os
from io import BytesIO

import streamlit as st
import torch
from PIL import Image
from sacred import Experiment

from frozen.datamodules.datamodule_base import get_pretrained_tokenizer
from frozen.models import LitFROZEN
from frozen.transforms import pixelbert_transform

ex = Experiment("demo")
VIS_MODE_DICT = {"GLOBAL+LOCAL": 'duel', "GLOBAL": 'global', "LOCAL": 'local'}


@ex.config
def config():
    lm = 'gpt2'
    emb_key = 'n_embd'
    image_size = 384


def convert_to_inputs(img_input, image_size):
    if isinstance(img_input, BytesIO):
        image = Image.open(img_input).convert('RGB')
    else:
        image = img_input
    tensor = pixelbert_transform(size=image_size)(image).unsqueeze(0)
    return image, tensor


@torch.no_grad()
def infer(model, img, text):
    tokens = model.encode(text)
    tokens = dict(
        input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0).to(model.device),
        attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0).to(model.device)
    )
    output = model.vqa_infer_once(img.cuda(), tokens, 10)
    output = model.decode(output)
    return output


@ex.automain
def main(
    lm,
    emb_key,
    image_size
):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.cuda.set_device(7)  # use last device
    state = st.session_state
    st.title('Frozen VQAv2 Demo')
    vis_mode = st.selectbox(
        'Choose a method to put visual token.',
        list(VIS_MODE_DICT.keys())
    )
    vis_mode = VIS_MODE_DICT[vis_mode]
    load_path = f'/project/FROZEN/FROZEN_{vis_mode}_pretrained.ckpt'
    with st.spinner("Changing model, please wait..."):
        if state.get(f'{vis_mode}_model') is None:
            model = LitFROZEN.from_pretrained(lm, emb_key=emb_key, vis_mode=vis_mode)
            checkpoint = torch.load(load_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            if state.get('tokenizer') is not None:
                tokenizer = state.tokenizer
            else:
                tokenizer = get_pretrained_tokenizer(lm, emb_key=emb_key, pad_token='<|endoftext|>')
            model.set_tokenizer(tokenizer)
            model.setup('test')
            model.eval().cuda()
            state[f'{vis_mode}_model'] = model
        else:
            model = state[f'{vis_mode}_model']
    with st.form('upload_image'):
        image = tensor = None
        file = st.file_uploader('Upload an image.')
        if file is None:
            if state.get('inputs') is not None:
                image, tensor = state.inputs
        else:
            image, tensor = convert_to_inputs(file, image_size)
            state.inputs = (image, tensor)
        input_text = st.text_input('Write a text to extend using visual information.')
        is_question = st.checkbox("Transform to question format")
        submit = st.form_submit_button('Submit!')
        if submit:
            if image is not None:
                st.image(image, caption='Input Image')
                with st.spinner("The model is thinking..."):
                    if is_question:
                        input_text = f"Question: {input_text} Answer:"
                    output = infer(model, tensor, input_text)
                if is_question:
                    st.caption(input_text.replace(' Answer:', ''))
                else:
                    st.caption(f'Input Text: {input_text}')
                st.write(f'Answer: {output}')
            else:
                st.write('Please upload an image to infer.')
    # st.button('Random Sampling from VQAv2 Dataset')
