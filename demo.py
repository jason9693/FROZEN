import json
import os
import random
import time
import uuid
from io import BytesIO

import numpy as np
import streamlit as st
import torch
from PIL import Image

from frozen.datamodules.datamodule_base import get_pretrained_tokenizer
from frozen.models import GPT2LitFROZEN, ElectraMaskedLitFROZEN, BertMaskedLitFROZEN
from frozen.profiles import VIS_MODE_DICT, LM_MODE_DICT
from frozen.transforms import pixelbert_transform


PYTHON_PATH = os.path.abspath('./')
DEMO_VIS_MODE_DICT = {
    'GLOBAL+LOCAL': dict(**VIS_MODE_DICT['duel']),
    'GLOBAL': dict(**VIS_MODE_DICT['global']),
    'LOCAL': dict(**VIS_MODE_DICT['local'])
}
DEMO_LM_MODE_DICT = {
    'GPT2': dict(**LM_MODE_DICT['gpt2']),
    'ELECTRA-BASE': dict(**LM_MODE_DICT['electra-base']),
    'BERT-BASE': dict(**LM_MODE_DICT['bert-base'])
}


def _get_model(lm_mode, vis_mode):
    path_key = f'{DEMO_LM_MODE_DICT[lm_mode]["demo_path"]}_{DEMO_VIS_MODE_DICT[vis_mode]["demo_path"]}'
    path = os.path.join(PYTHON_PATH, f'BiFROST_{path_key}.ckpt')
    hpath = DEMO_LM_MODE_DICT[lm_mode]['lm']
    emb_key = DEMO_LM_MODE_DICT[lm_mode]['emb_key']
    vis_mode = DEMO_VIS_MODE_DICT['mode']
    num_vis_tokens = DEMO_VIS_MODE_DICT['num_vis_tokens']
    model = DEMO_LM_MODE_DICT[lm_mode]['cls'].from_pretrained(
        hpath, emb_key=emb_key, vis_mode=vis_mode, num_vis_tokens=num_vis_tokens)
    state_dict = torch.load(path)['state_dict']
    model.load_state_dict(state_dict)
    return model


def convert_to_inputs(img_input, image_size):
    if isinstance(img_input, BytesIO):
        image = Image.open(img_input).convert('RGB')
    else:
        image = img_input
    image_tensor = pixelbert_transform(size=image_size)(image).unsqueeze(0)
    return image, image_tensor


@torch.no_grad()
def infer_and_export(
    model,
    image_tensor,
    image,
    input_text,
    is_question,
    use_random_masking,
    max_length,
    masking_rate,
    inference_method,
    info=None
):
    if inference_method == 'plm':
        plm_infer_and_export(model, image_tensor, image, input_text, is_question, max_length, info)
    else:
        mlm_infer_and_export(model, image_tensor, image, input_text, use_random_masking, masking_rate, info)


def plm_infer_and_export(model, image_tensor, image, input_text, is_question, max_length, info=None):
    if image is not None:
        with st.spinner('The model is thinking...'):
            s = time.time()
            if is_question:
                caption = f'Question: {input_text}'
                input_text = f'Question: {input_text} Answer:'
            else:
                caption = f'Input Text: {input_text}'
            tokens = model.encode(input_text)
            input_ids = tokens['input_ids']
            tokens_length = len(tokens['input_ids'])
            tokens = dict(
                input_ids=torch.tensor(input_ids).unsqueeze(0).to(model.device),
                attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0).to(model.device)
            )
            output_ids = output = model.zero_shot_infer(image_tensor.cuda(), tokens, max_length)
            output = model.decode(output)
            output_ids = output_ids.squeeze().data.cpu().numpy().tolist()
            e = time.time()
            elapsed_time = e-s
        st.caption(caption)
        st.write(f'Answer: {output}')
        with st.expander("More info..."):
            info_txt = [
                f"Inference Method: Causal Language Modeling",
                f"Inference Time: {elapsed_time:.3f}s"
            ]
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


def mlm_infer_and_export(model, image_tensor, image, input_text, use_random_masking, masking_rate, info=None):
    if image is not None:
        with st.spinner('The model is thinking...'):
            s = time.time()
            tokens = model.encode(input_text)
            input_ids = tokens['input_ids']
            indices = [i for i, input_id in enumerate(input_ids) if input_id == model.tokenizer.mask_token_id]
            if use_random_masking:
                indices += list(random.sample(range(1, len(input_ids)-1), int(masking_rate*len(input_ids))))
                indices.sort()
                input_ids = [model.tokenizer.mask_token_id if i in indices else iid for i, iid in enumerate(input_ids)]
            tokens_length = len(tokens['input_ids'])
            tokens = dict(
                input_ids=torch.tensor(input_ids).unsqueeze(0).to(model.device),
                attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0).to(model.device)
            )
            output_ids = model.zero_shot_infer(image_tensor.cuda(), tokens)
            output = [output_ids[idx] if idx in indices else iid for idx, iid in enumerate(input_ids)]
            output = output[1:-1]
            output = model.decode(output)
            output_ids = output_ids.squeeze().data.cpu().numpy().tolist()
            e = time.time()
            elapsed_time = e-s
        input_text = model.decode(input_ids)
        st.caption(f'Input Text: {input_text}')
        st.write(f'Answer: {output}')
        with st.expander("More info..."):
            info_txt = [
                f"Inference Method: Masked Language Modeling",
                f"Inference Time: {elapsed_time:.3f}s"
            ]
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
    selected_lm_mode = st.sidebar.selectbox('Choose a language model.', list(DEMO_LM_MODE_DICT.keys()))
    lm = DEMO_LM_MODE_DICT[selected_lm_mode]['lm']
    emb_key = DEMO_LM_MODE_DICT[selected_lm_mode]['emb_key']
    pad_token = DEMO_LM_MODE_DICT[selected_lm_mode].get('pad_token')
    image_size = 384
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.cuda.set_device(torch.cuda.device_count()-1)  # use last device
    state = st.session_state
    selected_vis_mode = st.sidebar.selectbox(
        'Choose a method to put visual token.',
        list(DEMO_VIS_MODE_DICT.keys())
    )
    info = {'Language Model': selected_lm_mode, 'Vision Token Type': selected_vis_mode, 'Input Image Size': image_size}
    path_key = '_'.join([DEMO_LM_MODE_DICT[selected_lm_mode]['demo_path'], DEMO_VIS_MODE_DICT[selected_vis_mode]['demo_path']])
    inference_method = DEMO_LM_MODE_DICT[selected_lm_mode]['inference_method']
    load_path = f'/project/FROZEN/FROZEN_{path_key}_pretrained.ckpt'
    if not os.path.exists(load_path):
        st.sidebar.error(f'Cannot load the model: {load_path}')
    else:
        with st.spinner('Loading the selected model, please wait...'):
            if state.get(f'{selected_lm_mode}_{selected_vis_mode}_MODEL') is None:
                model = _get_model(selected_lm_mode, selected_vis_mode)
                checkpoint = torch.load(load_path, map_location='cpu')
                model.load_state_dict(checkpoint['state_dict'])
                if state.get(f'{selected_lm_mode}_{selected_vis_mode}_TOKENIZER') is not None:
                    tokenizer = state[f'{selected_lm_mode}_{selected_vis_mode}_TOKENIZER']
                else:
                    tokenizer = get_pretrained_tokenizer(lm, emb_key=emb_key, pad_token=pad_token)
                    state[f'{selected_lm_mode}_{selected_vis_mode}_TOKENIZER'] = tokenizer
                model.set_tokenizer(tokenizer)
                model.setup('test')
                model.eval().cuda()
                state[f'{selected_lm_mode}_{selected_vis_mode}_MODEL'] = model
                state[f'{selected_lm_mode}_{selected_vis_mode}_INFERENCE_METHOD'] = inference_method
            else:
                model = state[f'{selected_lm_mode}_{selected_vis_mode}_MODEL']
                inference_method = state[f'{selected_lm_mode}_{selected_vis_mode}_INFERENCE_METHOD']
        reload_model = st.sidebar.button('Reload Model')
        if reload_model:
            if state.get(f'{selected_lm_mode}_{selected_vis_mode}_MODEL') is not None:
                del state[f'{selected_lm_mode}_{selected_vis_mode}_MODEL']
            st.experimental_rerun()
        if inference_method == 'plm':
            max_length = st.sidebar.select_slider('Set the maximum length of output sentence.', list(range(10, 31)))
            masking_rate = None
        else:
            max_length = None
            masking_rate = st.sidebar.select_slider(
                'Set the percentages of masked tokens.', list(np.linspace(0., 0.5, num=50)))
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
            input_text = st.sidebar.text_input('Write the text input.')
            if inference_method == 'plm':
                is_question = st.sidebar.checkbox('Transform to question format')
                use_random_masking = False
            else:
                is_question = False
                use_random_masking = st.sidebar.checkbox('Transform to random masked format')
            submitted = st.sidebar.button('Submit')
            if submitted and does_image_exist and input_text:
                infer_and_export(
                    model,
                    image_tensor,
                    image,
                    input_text,
                    is_question,
                    use_random_masking,
                    max_length,
                    masking_rate,
                    inference_method,
                    info
                )
            if does_image_exist and input_text:
                add_to_examples = st.button('Add to Examples')
                if add_to_examples:
                    with open(examples_path, 'r') as f:
                        examples = json.load(f)
                    image_dir = os.path.join(example_dir, 'images')
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f'{uuid.uuid4()}.png')
                    new_example = dict()
                    new_example['image'] = image_path
                    new_example['input_text'] = input_text
                    new_example['is_question'] = is_question
                    new_example['use_random_masking'] = use_random_masking
                    examples.append(new_example)
                    image.save(image_path)
                    with open(examples_path, 'w') as f:
                        json.dump(examples, f)
                    st.success("Your example is added!")
        elif submit_mode == submit_modes[1]:
            with open(examples_path, 'r') as f:
                examples = json.load(f)
            options = [f"{i}. {example['input_text']}" for i, example in enumerate(examples)]
            example = st.sidebar.selectbox('Choose an example set.', options)
            idx = options.index(example)
            image_path = examples[idx]['image']
            image = Image.open(image_path).convert('RGB')
            st.image(image, caption='Input Image')
            input_text = examples[idx]['input_text']
            is_question = examples[idx]['is_question']
            use_random_masking = examples[idx]['use_random_masking']
            _, image_tensor = convert_to_inputs(image, image_size)
            infer_and_export(
                model,
                image_tensor,
                image,
                input_text,
                is_question,
                use_random_masking,
                max_length,
                masking_rate,
                inference_method,
                info
            )
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