import glob
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
from einops import rearrange

from frozen.datamodules.datamodule_base import get_pretrained_tokenizer
from frozen.models import BiFrostBase
from frozen.transforms import pixelbert_transform

PYTHON_PATH = os.path.abspath('./')


def get_all_checkpoint_paths(root_dir=os.path.join(PYTHON_PATH, 'BiFrost'), path_format='BiFrost*.ckpt'):
    return list(glob.glob(os.path.join(root_dir, path_format)))


def convert_to_inputs(img_input, image_size):
    if isinstance(img_input, BytesIO):
        image = Image.open(img_input).convert('RGB')
    else:
        image = img_input
    image_tensor = pixelbert_transform(size=image_size)(image).unsqueeze(0)
    return image, image_tensor


def load_and_profile(path):
    return BiFrostBase.from_bifrost_checkpoint(path, config_dict=dict(output_attentions=True))


def visualize_attentions(
    image,
    vis_mode,
    num_global_tokens,
    local_output_size,
    attentions,
    input_tokens
):
    with st.expander('Attention Maps'):
        h, w = local_output_size, int(local_output_size*1333./800.)
        if vis_mode == 'duel':
            start_idx = num_global_tokens
            end_idx = h*w+num_global_tokens
            num_vis_tokens = num_global_tokens+h*w
        elif vis_mode == 'local':
            start_idx = 0
            end_idx = h*w
            num_vis_tokens = h*w
        else:
            return
        is_sum_all_layers = st.checkbox('Sum all layers')
        if not is_sum_all_layers:
            layer_idx = st.select_slider(
                'Select a layer index.',
                options=list(range(len(attentions))),
                value=len(attentions)-1
            )
            attn = attentions[layer_idx][0]
            num_heads = attn.size(0)
        else:
            attn = sum(attentions)[0]
            num_heads = attn.size(0)
        is_sum_all_heads = st.checkbox('Sum all heads')
        if not is_sum_all_heads:
            head_idx = st.select_slider('Select a head index.', options=list(range(num_heads)))
            attn = attn[head_idx]
        else:
            attn = attn.sum(dim=0)
        attn = attn[num_vis_tokens:, start_idx:end_idx]
        attn = rearrange(attn, 'n (h w) -> n h w', h=h, w=w)
        word_idx = st.select_slider(
            'Select a word to show attention.',
            options=list(range(len(input_tokens))),
            format_func=lambda idx: input_tokens[idx]
        )
        heatmap = attn[word_idx]
        # heatmap = (heatmap-heatmap.mean())/heatmap.std()
        # heatmap = torch.clamp(heatmap, 1.0, 3.0)
        heatmap = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min())
        heatmap = heatmap.data.cpu().numpy()
        _w, _h = image.size
        overlay = Image.fromarray(np.uint8(heatmap*255), "L").resize((_w, _h), resample=Image.NEAREST)
        image_rgba = image.copy()
        image_rgba.putalpha(overlay)
        st.image(image_rgba, 'Attention Map')


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
    config,
    info=None
):
    if model.INFERENCE_METHOD == 'plm':
        plm_infer_and_export(model, image_tensor, image, input_text, is_question, max_length, config, info)
    else:
        mlm_infer_and_export(model, image_tensor, image, input_text, use_random_masking, masking_rate, config, info)


def plm_infer_and_export(model, image_tensor, image, input_text, is_question, max_length, config, info=None):
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
            output, attentions = model.infer(image_tensor.cuda(), tokens, max_length)
            output_ids = output
            output = model.decode(output)
            output_ids = output_ids.squeeze().data.cpu().numpy().tolist()
            e = time.time()
            elapsed_time = e-s
        st.caption(caption)
        st.write(f'Answer: {output}')
        input_tokens = model.tokenizer.convert_ids_to_tokens(input_ids)
        output_tokens = model.tokenizer.convert_ids_to_tokens(output_ids)
        if isinstance(output_tokens, list):
            inferred_tokens = input_tokens+output_tokens[:-1]
        else:
            inferred_tokens = input_tokens
        if config['vis_mode'] in ('duel', 'local'):
            visualize_attentions(
                image,
                config['vis_mode'],
                2,
                4,
                attentions,
                inferred_tokens
            )
        with st.expander("More info..."):
            info_txt = [
                f"Inference Time: {elapsed_time:.3f}s"
            ]
            if info is not None:
                info_txt += [f"{k}: {v}" for k, v in info.items()]
            info_txt.append(f'Input Vision Token Length: {model.vis_model.num_tokens}')
            info_txt.append(f'Input Language Token Length: {tokens_length}')
            info_txt.append(f'Tokenized Input Text: {input_tokens}')
            info_txt.append(f'Raw Input Ids Info: {input_ids}')
            info_txt.append(f'Raw Output Ids Info: {output_ids}')
            for s in info_txt:
                st.write(s)
    else:
        st.write('Please upload an image to infer.')


def mlm_infer_and_export(model, image_tensor, image, input_text, use_random_masking, masking_rate, config, info=None):
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
            output, attentions = model.infer(image_tensor.cuda(), tokens)
            output_ids = output
            output = [output_ids[idx] if idx in indices else iid for idx, iid in enumerate(input_ids)]
            output = output[1:-1]
            output = model.decode(output)
            output_ids = output_ids.squeeze().data.cpu().numpy().tolist()
            e = time.time()
            elapsed_time = e-s
        input_text = model.decode(input_ids)
        st.caption(f'Input Text: {input_text}')
        st.write(f'Answer: {output}')
        input_tokens = model.tokenizer.convert_ids_to_tokens(input_ids)[1:-1]
        if config['vis_mode'] in ('duel', 'local'):
            visualize_attentions(
                image,
                config['vis_mode'],
                2,
                4,
                attentions,
                model.tokenizer.convert_ids_to_tokens(input_ids)
            )
        with st.expander("More info..."):
            info_txt = [
                f"Inference Time: {elapsed_time:.3f}s"
            ]
            if info is not None:
                info_txt += [f"{k}: {v}" for k, v in info.items()]
            info_txt.append(f'Input Vision Token Length: {model.vis_model.num_tokens}')
            info_txt.append(f'Input Language Token Length: {tokens_length}')
            info_txt.append(f'Tokenized Input Text: {input_tokens}')
            info_txt.append(f'Raw Input Ids Info: {input_ids}')
            info_txt.append(f'Raw Output Ids Info: {output_ids}')
            for s in info_txt:
                st.write(s)
    else:
        st.write('Please upload an image to infer.')


def main():
    st.sidebar.title('BiFrost Demo')
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
                model, config, tokenizer = state[load_path]
            else:
                model, config = load_and_profile(load_path)
                checkpoint = torch.load(load_path, map_location='cpu')
                model.load_state_dict(checkpoint['state_dict'])
                tokenizer = get_pretrained_tokenizer(
                    config['hface_path'], emb_key=config['emb_key'], pad_token=config.get('pad_token'))
                model.set_tokenizer(tokenizer)
                model.setup('test')
                model.eval().cuda()
                state[load_path] = (model, config, tokenizer)
        info = {
            'Language Model': model.LANGUAGE_MODEL,
            'Vision Token Type': config['vis_mode'],
            'Input Image Size': image_size,
            'Checkpoint Path': load_path
        }
        reload_model = st.sidebar.button('Reload Model')
        if reload_model:
            del state[load_path]
            st.experimental_rerun()
        if model.INFERENCE_METHOD == 'plm':
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
            if model.INFERENCE_METHOD == 'plm':
                is_question = st.sidebar.checkbox('Transform to question format')
                use_random_masking = False
            else:
                is_question = False
                use_random_masking = st.sidebar.checkbox('Transform to random masked format')
            if does_image_exist and input_text:
                infer_and_export(
                    model,
                    image_tensor,
                    image,
                    input_text,
                    is_question,
                    use_random_masking,
                    max_length,
                    masking_rate,
                    config,
                    info
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
                    new_example['input_text'] = input_text
                    new_example['is_question'] = is_question
                    new_example['use_random_masking'] = use_random_masking
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
            input_text = examples[idx]['input_text']
            is_question = examples[idx]['is_question']
            use_random_masking = examples[idx]['use_random_masking']
            _, image_tensor = convert_to_inputs(image, image_size)
            input_text = st.sidebar.text_input('Write the text input.', value=input_text)
            infer_and_export(
                model,
                image_tensor,
                image,
                input_text,
                is_question,
                use_random_masking,
                max_length,
                masking_rate,
                config,
                info
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
