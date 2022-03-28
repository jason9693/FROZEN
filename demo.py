import json
import os
import time
from io import BytesIO

import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer

from frozen.datamodules.datamodule_base import get_pretrained_tokenizer
from frozen.models import GPT2LitFROZEN, ElectraLitFROZEN
from frozen.transforms import pixelbert_transform

VIS_MODE_DICT = {
    'GLOBAL+LOCAL': {'model_key': 'duel', 'path': 'duel'},
    'GLOBAL': {'model_key': 'global', 'path': 'global'},
    'LOCAL': {'model_key': 'local', 'path': 'local'},
    'GLOBAL+LOCAL(FINETUNED)': {'model_key': 'duel', 'path': 'duel_finetune'},
    'GLOBAL(FINETUNED)': {'model_key': 'global', 'path': 'global_finetune'},
    'LOCAL(FINETUNED)': {'model_key': 'local', 'path': 'local_finetune'},
    'GLOBAL+LOCAL(FROZEN+FINETUNED)': {'model_key': 'duel', 'path': 'duel_frozen'},
    'GLOBAL(FROZEN+FINETUNED)': {'model_key': 'global', 'path': 'global_frozen'},
    # 'LOCAL(FROZEN+FINETUNED)': {'model_key': 'local', 'path': 'local_frozen'},
}
LM_MODE_DICT = {
    'GPT2': {'lm': 'gpt2', 'emb_key': 'n_embd', 'pad_token': '<|endoftext|>', 'path': 'gpt2'},
    'ELECTRA-SMALL': {'lm': 'google/electra-small-discriminator', 'emb_key': 'embedding_size', 'path': 'electra_small'},
    'ELECTRA-BASE': {'lm': 'google/electra-base-discriminator', 'emb_key': 'embedding_size', 'path': 'electra_base'}
}


def convert_to_inputs(img_input, image_size):
    if isinstance(img_input, BytesIO):
        image = Image.open(img_input).convert('RGB')
    else:
        image = img_input
    tensor = pixelbert_transform(size=image_size)(image).unsqueeze(0)
    return image, tensor


@torch.no_grad()
def infer(model, img, text, max_length):
    tokens = model.encode(text)
    input_ids = tokens['input_ids']
    tokens_length = len(tokens['input_ids'])
    tokens = dict(
        input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0).to(model.device),
        attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0).to(model.device)
    )
    output_ids = output = model.zero_shot_infer(img.cuda(), tokens, max_length)
    output = model.decode(output)
    output_ids = output_ids.squeeze().data.cpu().numpy().tolist()
    return output, tokens_length, input_ids, output_ids


def infer_and_export(model, tensor, image, input_text, is_question, max_length, info=None):
    if image is not None:
        with st.spinner('The model is thinking...'):
            if is_question:
                text = f'Question: {input_text} Answer:'
            else:
                text = input_text
            s = time.time()
            output, tokens_length, input_ids, output_ids = infer(model, tensor, text, max_length)
            e = time.time()
            elapsed_time = e-s
        if is_question:
            st.caption(f'Question: {input_text}')
        else:
            st.caption(f'Input Text: {input_text}')
        st.write(f'Answer: {output}')
        with st.expander("More info..."):
            info_txt = [f"Inference Time: {elapsed_time:.3f}s"]
            if info is not None:
                info_txt += [f"{k}: {v}" for k, v in info.items()]
            info_txt.append(f'Input Vision Token Length: {model.v_encoder.num_tokens}')
            info_txt.append(f'Input Language Token Length: {tokens_length}')
            info_txt.append(f'Raw Input Ids Info: {input_ids}')
            info_txt.append(f'Raw Output Ids Info: {output_ids}')
            for s in info_txt:
                st.write(s)
    else:
        st.write('Please upload an image to infer.')


def main():
    st.sidebar.title('BiFrost VQAv2 Demo')
    selected_lm_mode = st.sidebar.selectbox('Choose a language model.',
                                            list(LM_MODE_DICT.keys()))
    lm = LM_MODE_DICT[selected_lm_mode]['lm']
    emb_key = LM_MODE_DICT[selected_lm_mode]['emb_key']
    pad_token = LM_MODE_DICT[selected_lm_mode].get('pad_token')
    image_size = 384
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.cuda.set_device(torch.cuda.device_count()-1)  # use last device
    state = st.session_state
    selected_vis_mode = st.sidebar.selectbox(
        'Choose a method to put visual token.',
        list(VIS_MODE_DICT.keys())
    )
    info = {'Language Model': selected_lm_mode, 'Vision Token Type': selected_vis_mode, 'Input Image Size': image_size}
    path_key = '_'.join([LM_MODE_DICT[selected_lm_mode]['path'], VIS_MODE_DICT[selected_vis_mode]['path']])
    load_path = f'/project/FROZEN/FROZEN_{path_key}_pretrained.ckpt'
    if not os.path.exists(load_path):
        st.sidebar.error(f'Cannot load the model: {load_path}')
    else:
        vis_mode = VIS_MODE_DICT[selected_vis_mode]['model_key']
        with st.spinner('Loading the selected model, please wait...'):
            if state.get(f'{selected_lm_mode}_{selected_vis_mode}_MODEL') is None:
                if 'gpt' in lm:
                    model = GPT2LitFROZEN.from_pretrained(lm, emb_key=emb_key, vis_mode=vis_mode)
                elif 'electra' in lm:
                    model = ElectraLitFROZEN.from_pretrained(lm, emb_key=emb_key, vis_mode=vis_mode)
                else:
                    raise ValueError
                checkpoint = torch.load(load_path, map_location='cpu')
                model.load_state_dict(checkpoint['state_dict'])
                if state.get(f'{selected_lm_mode}_{selected_vis_mode}_TOKENIZER') is not None:
                    tokenizer = state[f'{selected_lm_mode}_{selected_vis_mode}_TOKENIZER']
                else:
                    if 'electra' in lm:
                        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                    else:
                        tokenizer = get_pretrained_tokenizer(lm, emb_key=emb_key, pad_token=pad_token)
                    state[f'{selected_lm_mode}_{selected_vis_mode}_TOKENIZER'] = tokenizer
                model.set_tokenizer(tokenizer)
                model.setup('test')
                model.eval().cuda()
                state[f'{selected_lm_mode}_{selected_vis_mode}_MODEL'] = model
            else:
                model = state[f'{selected_lm_mode}_{selected_vis_mode}_MODEL']
        reload_model = st.sidebar.button('Reload Model')
        if reload_model:
            if state.get(f'{selected_lm_mode}_{selected_vis_mode}_MODEL') is not None:
                del state[f'{selected_lm_mode}_{selected_vis_mode}_MODEL']
            st.experimental_rerun()
        max_length = st.sidebar.select_slider('Set the maximum length of output sentence.', list(range(10, 31)))
        example_dir = os.path.join(os.getcwd(), 'examples')
        os.makedirs(example_dir, exist_ok=True)
        examples_path = os.path.join(example_dir, 'examples.json')
        if not os.path.exists(examples_path):
            with open(examples_path, 'w') as f:
                json.dump([], f)
        inference_modes = ['Upload your own image', 'Sample from examples']
        inference_mode = st.sidebar.selectbox('', inference_modes)
        if inference_mode == inference_modes[0]:
            image = tensor = None
            file = st.sidebar.file_uploader('Upload an image.')
            render = False
            if file is None:
                if state.get('inputs') is not None:
                    image, tensor = state.inputs
                    render = True
            else:
                image, tensor = convert_to_inputs(file, image_size)
                state.inputs = (image, tensor)
                render = True
            if render:
                st.image(image, caption='Input Image')
            input_text = st.sidebar.text_input('Write the text to extend using visual information.')
            is_question = st.sidebar.checkbox('Transform to question format')
            if render and input_text:
                infer_and_export(model, tensor, image, input_text, is_question, max_length, info)
                add_to_examples = st.button('Add to Examples')
                if add_to_examples:
                    with open(examples_path, 'r') as f:
                        examples = json.load(f)
                    image_dir = os.path.join(example_dir, 'images')
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f'example_{len(examples)}.png')
                    new_example = dict()
                    new_example['image'] = image_path
                    new_example['input_text'] = input_text
                    new_example['is_question'] = is_question
                    examples.append(new_example)
                    image.save(image_path)
                    with open(examples_path, 'w') as f:
                        json.dump(examples, f)
                    st.success("Your example is added!")
        elif inference_mode == inference_modes[1]:
            with open(examples_path, 'r') as f:
                examples = json.load(f)
            options = [f"{i}. {example['input_text']}" for i, example in enumerate(examples)]
            example = st.sidebar.selectbox('Choose an example set.', options)
            idx = options.index(example)
            image = Image.open(examples[idx]['image']).convert('RGB')
            st.image(image, caption='Input Image')
            input_text = examples[idx]['input_text']
            is_question = examples[idx]['is_question']
            _, tensor = convert_to_inputs(image, image_size)
            infer_and_export(model, tensor, image, input_text, is_question, max_length, info)
            delete = st.sidebar.button('Delete this example')
            if delete:
                del examples[idx]
                st.sidebar.error("Example is deleted.")
                with open(examples_path, 'w') as f:
                    json.dump(examples, f)
                st.experimental_rerun()


if __name__ == "__main__":
    main()
