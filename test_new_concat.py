import os
import re
from html import unescape

import clip
from transformers import AutoTokenizer
import numpy as np

from typing import Tuple, List, Union, Optional
from tqdm import tqdm, trange
import skimage.io as io
import PIL.Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle
import torch
import json
import wandb

import spacy
from bs4 import BeautifulSoup

from model_new_concat import ClipCaptionPrefix, ClipCaptionModel


run_wandb = True
annotation_setting = 'sis'
language_model = 'gpt2-xl'
#model_path = '/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/checkpoints/wed1206/vist_prevMN_RN50x4_context_num2_train_20_10_contras_gpt2l_newcon_num2-004.pt'
model_path = '/scratch/song0018/vist_data/checkpoints/tue0227/vist_prevMN_RN50x4_context_num1_train_32_32_contras_gpt2xl_newcon_num1_gptnce-000.pt'


is_rn = True
MAPPING_TYPE = 'transformer'
normalize_prefix = True
text_first = False
clip_model_type = "RN50x4" if is_rn else "ViT-B_32"
#out_path = f'/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/results/preds_{annotation_setting}_{clip_model_type}_{MAPPING_TYPE}_1206_contras_gpt2l_newcon_num2-004.pkl'
out_path = f'/scratch/song0018/vist_data/results/preds_{annotation_setting}_{clip_model_type}_{MAPPING_TYPE}_0227_contras_gpt2xl_newcon_num1_gptnce-cont000.pkl'

NUM_LAYERS = 8
prefix_length = 32
CLIP_LENGTH = 32

split = 'test'
PREFIX_SIZE = 640*2 if is_rn else 512*2
img_dir = f"/scratch/song0018/vist_data/images/{split}_images"
annotations = f"/scratch/song0018/vist_data/annotations/{annotation_setting}"
#
# img_dir = "/hpc/shared/uu_vl/yingjin_datasets/vist_data/images/test_images"
# annotations = f"/hpc/shared/uu_vl/yingjin_datasets/vist_data/annotations/{annotation_setting}"
# img_embeds_fn = f"/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/{annotation_setting}_img_embeds_{clip_model_type}_{split}.pkl"


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# with open(img_embeds_fn, 'rb') as f:
#     img_embeds = pickle.load(f)

clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

if run_wandb == True:
    wandb.init(project='CLIP_prefix_CAP-inference',
               config={
                      "lm": language_model,
                      "loaded_model": model_path,
                   "output_path": out_path,
               })


tokenizer = AutoTokenizer.from_pretrained(language_model, token = hf_token)

if annotation_setting == "dii":
    info = json.load(open(os.path.join(annotations, f"{split}.description-in-isolation.json")))
elif annotation_setting == "sis":
    info = json.load(open(os.path.join(annotations, f"{split}.story-in-sequence.json")))
else:
    raise Exception("Annotation json file is not found.")





N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]


D = torch.device
CPU = torch.device('cpu')

def clean_strs(text):
    # Regular expression pattern to match URLs
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Replace URLs with an empty string
    cleaned_text = re.sub(pattern, '', text)
    # Remove HTML tags
    cleaned_text= re.sub('<.*?>', '', cleaned_text)
    # Remove HTML symbols
    cleaned_text = re.sub('&[a-zA-Z0-9]+;', '', cleaned_text)
    # Decode HTML entities
    cleaned_text = unescape(cleaned_text)
    # Remove numbers and symbols
    cleaned_text = re.sub('[&@]', '', cleaned_text)

    return cleaned_text

def add_end_marks(text):
    text = text.rstrip()
    if not text.endswith('.') and not text.endswith('?') and not text.endswith('!'):
        text += '.'
    return text

# def get_device(device_id: int) -> D:
#     if not torch.cuda.is_available():
#         return CPU
#     device_id = min(torch.cuda.device_count() - 1, device_id)
#     #device_id = torch.cuda.device_count() - 1
#     return torch.device(f'cuda:{device_id}')
#
#
# CUDA = get_device




model = ClipCaptionPrefix(prefix_length, clip_length=CLIP_LENGTH, prefix_size=PREFIX_SIZE,
                          num_layers=NUM_LAYERS, mapping_type = MAPPING_TYPE)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.eval()
model = model.to(device)


def generate_sentence_context(decoding_mode, prev_sent, prefix):
    # #prev_sent = tokenizer.bos_token + prev_sent + tokenizer.eos_token
    # prev_tokens = torch.tensor(tokenizer.encode(prev_sent)).unsqueeze(0).to(device)
    # # context_text = model.gpt.transformer.wte(prev_tokens)
    # context_text = model.lm.get_input_embeddings()(prev_tokens)
    # prev_context = torch.cat([bos_token, context_text, eos_token], dim=1)
    # if text_first:
    #     concat_embed = torch.cat([prev_context, prefix_embed, bos_gener], dim=1)
    # else:
    #     concat_embed = torch.cat([prefix_embed, prev_context, bos_gener], dim=1)

    prev_tokens = clip_model.encode_text(clip.tokenize(prev_sent.strip(), context_length=77, truncate=True).to(device))
    prev_tokens = prev_tokens.float()
    prev_tokens /= prev_tokens.norm(dim=-1, keepdim=True)

    concat_embed = torch.cat((prefix, prev_tokens), dim=1)

    concat_embed = model.clip_project(concat_embed).reshape(1, prefix_length, -1)

    if decoding_mode == "beam":
        sample_outputs = model.gpt.generate(
            inputs_embeds=concat_embed,
            max_length=30,
            num_beams=5,
            num_return_sequences=3,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(sample_outputs[1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    elif decoding_mode == "topk":
        sample_outputs = model.gpt.generate(
            inputs_embeds=concat_embed,
            do_sample=True,
            max_length=30,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    elif decoding_mode == "nucleus":
        sample_outputs = model.gpt.generate(
            inputs_embeds=concat_embed,
            do_sample=True,
            max_length=30,
            top_p=0.9,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    elif decoding_mode == "contrastive":
        sample_outputs = model.gpt.generate(
            inputs_embeds=concat_embed,
            do_sample=True,
            max_length=30,
            penalty_alpha=0.85,
            top_k=5,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    else:
        raise Exception("Decoding mode is not defined.")
    return generated_text



def predict_single_caption_context(img_fn, prev_beam, prev_topk, prev_nucleus, prev_contras):
    image = io.imread(img_dir + "/" + str(img_fn) + '.jpg')
    pil_image = PIL.Image.fromarray(image)
    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        if normalize_prefix:
            prefix = prefix / prefix.norm(2, -1).item()

        generated_text_prefix_beam = generate_sentence_context("beam", prev_beam, prefix)
        generated_text_prefix_topk = generate_sentence_context("topk", prev_topk, prefix)
        generated_text_prefix_nucleus = generate_sentence_context("nucleus", prev_nucleus, prefix)
        generated_text_prefix_contrastive = generate_sentence_context("contrastive", prev_contras, prefix)

    return generated_text_prefix_beam, generated_text_prefix_topk, generated_text_prefix_nucleus, generated_text_prefix_contrastive

if annotation_setting == "dii":
    dii = {'images': [], 'albums': [], 'annotations': []}
    for album in info['albums']:
        album['split'] = split
    dii['albums'] += info['albums']
    dii['images'] += info['images']
    dii['annotations'] += info['annotations']
    data = dii['annotations']
    print("%0d captions (DII) loaded from json " % len(data))
else:
    sis = {'images': [], 'albums': [], 'annotations': []}
    sis['albums'] += info['albums']
    sis['images'] += info['images']
    sis['annotations'] += info['annotations']

    data = sis['annotations']
    print("%0d captions (individual SIS) loaded from json " % len(data))



    album_info = {album['id']: album for album in sis['albums']}







available_data = []
for i in tqdm(range(len(data))):
    d = data[i][0]
    img_id = d['photo_flickr_id']
    # filename_train = f"/hpc/shared/uu_vl/yingjin_datasets/vist_data/images/resized_images/train/{int(img_id)}.jpg"
    # filename_test = f"/hpc/shared/uu_vl/yingjin_datasets/vist_data/images/resized_images/test/{int(img_id)}.jpg"
    filename_train = f"/scratch/song0018/vist_data/images/train_images/{int(img_id)}.jpg"
    filename_test = f"/scratch/song0018/vist_data/images/test_images/{int(img_id)}.jpg"

    if os.path.isfile(filename_train) or os.path.isfile(filename_test):
        available_data.append(d)

print("%0d captions with available images loaded from json " % len(available_data))
print(f"Generated sentences will be saved to {out_path}")


stories_dict = {}
for item in available_data:
    story_id = item['story_id']
    if story_id not in stories_dict.keys():
        stories_dict[story_id] = []
    stories_dict[story_id].append(item)

available_stories_dict = {}
for story_id in stories_dict.keys():
    if len(stories_dict[story_id]) == 5:
        available_stories_dict[story_id] = stories_dict[story_id]


test_results = {}

index = 0
for story_id in tqdm(available_stories_dict.keys()):
    print(story_id)

    story = available_stories_dict[story_id]
    true_story = [sent['original_text'] for sent in story]

    assert len(story) == 5

    beam_story = []
    topk_story = []
    nucleus_story = []
    contrastive_story = []
    img_ids = []
    for i in range(len(story)):
        sent = story[i]
        img_id = sent['photo_flickr_id']
        img_ids.append(img_id)
        true_text = sent['original_text']
        story_id = sent['story_id']
        image_order = sent['worker_arranged_photo_order']
        album_id = sent['album_id']

        assert str(image_order) == str(i)

        #start_sent = add_end_marks(album_info[album_id]['title'].strip() + ". " + clean_strs(album_info[album_id]['description']).strip()) + tokenizer.eos_token
        # start_sent = tokenizer.bos_token+ album_info[album_id]['title'].strip() + "." + tokenizer.eos_token

        # Clean the description from html formats
        album_info[album_id]['description'] = BeautifulSoup(album_info[album_id]['description'], 'html.parser').get_text().replace('\n', ' ')

        # Extract the first sentence of the discription to avoid too long text
        nlp = spacy.load('en_core_web_sm')
        album_info[album_id]['description'] = list(nlp(album_info[album_id]['description']).sents)[0].text.strip() if len(
            album_info[album_id]['description'].split()) > 50 else album_info[album_id]['description']

        start_sent = album_info[album_id]['title'].strip() + ". "+  album_info[album_id]['description'].strip()


        #filename = f"/hpc/shared/uu_vl/yingjin_datasets/vist_data/images/resized_images/test/{int(img_id)}.jpg"
        filename = f"/scratch/song0018/vist_data/images/{split}_images/{int(img_id)}.jpg"
        if not os.path.isfile(filename):
            #filename = f"/hpc/shared/uu_vl/yingjin_datasets/vist_data/images/resized_images/train/{int(img_id)}.jpg"
            filename = f"/scratch/song0018/vist_data/images/test_images/{int(img_id)}.jpg"
            if not os.path.isfile(filename):
                print(f"Image {img_id} not found !!!!!!")
                continue
        else:
            if i == 0:
                print(start_sent)
                beam_prediction, topk_prediction, nucleus_prediction, contrastive_prediction = predict_single_caption_context(img_id, start_sent, start_sent, start_sent, start_sent)
                beam_story.append(beam_prediction)
                topk_story.append(topk_prediction)
                nucleus_story.append(nucleus_prediction)
                contrastive_story.append(contrastive_prediction)
            else:
                beam_prediction, topk_prediction, nucleus_prediction, contrastive_prediction = predict_single_caption_context(img_id, prev_beam=beam_story[i-1], prev_topk=topk_story[i-1], prev_nucleus=nucleus_story[i-1], prev_contras=contrastive_story[i-1])
                beam_story.append(beam_prediction)
                topk_story.append(topk_prediction)
                nucleus_story.append(nucleus_prediction)
                contrastive_story.append(contrastive_prediction)

    print('story_id:', story_id)
    print('images:', img_ids)
    print("true:", true_story)
    print("beam:", beam_story)
    print("topk:", topk_story)
    print('nucleus:', nucleus_story)
    print('contrastive:', contrastive_story)
    print("====================================")
    test_results[story_id] = {"images": img_ids, "true": true_story, "beam": beam_story,
                              "topk": topk_story, "nucleus": nucleus_story, "contrastive": contrastive_story}

    if run_wandb == True:
        wandb.log({"story_id": story_id, "images": img_ids, "true": true_story,
                   "beam": beam_story, "topk": topk_story,
                   "nucleus": nucleus_story, "contrastive": contrastive_story})
    index += 1

    if (index + 1) % 500 == 0:
        with open(out_path, 'wb') as f:
            pickle.dump(test_results, f)

with open(out_path, 'wb') as f:
    pickle.dump(test_results, f)



