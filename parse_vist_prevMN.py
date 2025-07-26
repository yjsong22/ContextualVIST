import os

import numpy as np
import io

import torch

import clip
import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle
import json
from tqdm import tqdm
import argparse
import string
import spacy
from bs4 import BeautifulSoup



def main(clip_model_type: str, annot_dir:str, image_dir:str, context_num:int):
    annotation_setting = 'sis'
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)


    for split in ["val", "test", "train"]:
    #for split in ["train"]:
        print(" ====================== Processing "+ split.upper()+ " with context_num:"+str(context_num)+" ====================== ")

        if annot_dir.find("scratch") != -1:
            out_path = f"/scratch/song0018/vist_data/clip_embeds/vist_prevMN_{clip_model_name}_context_num{context_num}_{split}.pkl"
            sub_train_dir = "train_images"
        else:
            out_path = f"/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/vist_prevMN_{clip_model_name}_context_num{context_num}_{split}.pkl"
            sub_train_dir = "all_images"

        info = json.load(open(os.path.join(annot_dir, annotation_setting, split + ".story-in-sequence.json")))

        sis = {'images': [], 'albums': [], 'annotations': []}
        for album in info['albums']:
            album['split'] = split
        sis['albums'] += info['albums']
        sis['images'] += info['images']
        sis['annotations'] += info['annotations']

        # print(sis['albums'][10:20])
        # print(type(sis['images']))
        # sys.exit(0)

        data = sis['annotations']
        print("%0d captions (in SIS) loaded from json " % len(data))


        not_available_imgs = []
        for i in tqdm(range(len(data))):
            d = data[i][0]
            img_id = d['photo_flickr_id']
            filename_train = f"{image_dir}/{sub_train_dir}/{int(img_id)}.jpg"
            filename_test = f"{image_dir}/test_images/{int(img_id)}.jpg"
            filename_train_png = f"{image_dir}/{sub_train_dir}/{int(img_id)}.png"
            filename_test_png = f"{image_dir}/test_images/{int(img_id)}.png"
            if not os.path.isfile(filename_train) and not os.path.isfile(filename_test) and not os.path.isfile(filename_train_png) and not os.path.isfile(filename_test_png):
                not_available_imgs.append(img_id)

        print(f"{len(not_available_imgs)} images not found")

        sents = []
        for ann in data:
            sent = ann.copy()[0]
            sent['storylet_id'] = sent.pop('storylet_id')
            sent['order'] = sent.pop('worker_arranged_photo_order')
            sent['img_id'] = sent.pop('photo_flickr_id')
            sent['length'] = len(sent['text'].split())  # add length property

            for album in sis['albums']:
                if sent['album_id'] == album['id']:
                    sent['title'] = album['title']
                    if album['description'] != None and album['description'] != '' and album['description'] != ' ':
                        album['description'] += '.' if album['description'][-1] not in string.punctuation else ''
                    sent['description'] = album['description']
                    break

            # sent['title']  = next((item['title'] for item in sis['albums'] if item['id'] == sent['album_id']), None)
            # sent['description'] = next((item['description'] for item in sis['albums'] if item['id'] == sent['album_id']), None)
            sents += [sent]
        # print(sents[10])
        # print("==============================================")
        # print(sents[100])
        # print("==============================================")
        # print(sents[300])
        # print("==============================================")
        # print(sents[500])
        # print("==============================================")
        # print(sents[700: 710])
        # print("==============================================")
        # print(sents[1300])
        # print("==============================================")
        # sys.exit(0)

        story_ids = [annot['story_id'] for annot in sents]
        story_ids = list(set(story_ids))

        sis_stories = {}

        for story_id in tqdm(story_ids):
            sis_sents = {}
            img_ids = {}
            annot_ids = {}

            for sent in sents:
                if sent['story_id'] == story_id:
                    assert sent['tier'] == 'story-in-sequence'
                    img_ids[sent['img_id']] = sent['order']
                    sis_sents[sent['original_text']] = sent['order']
                    annot_ids[sent['storylet_id']] = sent['order']

            if bool(set(img_ids.keys()) & set(not_available_imgs)):
                continue
            else:
                assert len(img_ids.keys()) == 5
                assert len(sis_sents.keys()) == 5
                assert len (annot_ids.keys()) == 5

                sis_stories[story_id] = {
                'img_ids': list({k: v for k, v in sorted(img_ids.items(), key=lambda item: item[1])}.keys()),
                'sis_sents': list({k: v for k, v in sorted(sis_sents.items(), key=lambda item: item[1])}.keys()),
                'storylet_ids': list({k: v for k, v in sorted(annot_ids.items(), key=lambda item: item[1])}.keys()),
                'worker_arranged_photo_orders': [0, 1, 2, 3, 4],
                'story_id': story_id,
                }



        print(len(sents))
        print(len(story_ids))
        print(len(sis_stories))
        assert len(sents) == 5 * len(story_ids)

        for story_id, item in sis_stories.items():
            img_id = item['img_ids'][0]
            for sent in sents:
                if sent['img_id'] == img_id:
                    item['title'] = sent['title']
                    item['description'] = sent['description']
                    break

        all_image_embeddings = []
        all_context_embeddings = []
        all_captions = []
        clip_embedding_index = 0

        for i in tqdm(range(len(sis_stories))):
            story_id = list(sis_stories)[i]
            d = sis_stories[story_id]
            d['story_id'] = story_id
            img_ids = d['img_ids']

            img_path_batch = []
            for img_id in img_ids:
                filename_train = f"{image_dir}/{sub_train_dir}/{int(img_id)}.jpg"
                filename_test = f"{image_dir}/test_images/{int(img_id)}.jpg"
                filename_train_png = f"{image_dir}/{sub_train_dir}/{int(img_id)}.png"
                filename_test_png = f"{image_dir}/test_images/{int(img_id)}.png"
                if os.path.isfile(filename_train):
                    filename = filename_train
                elif os.path.isfile(filename_test):
                    filename = filename_test
                elif os.path.isfile(filename_train_png):
                    filename = filename_train_png
                elif os.path.isfile(filename_test_png):
                    filename = filename_test_png
                else:
                    if not os.path.isfile(filename_train) and os.path.isfile(filename_test) and not os.path.isfile(filename_train_png) and not os.path.isfile(filename_test_png):
                        continue
                    else:
                        raise Exception("Available images filter does not work!")

                img_path_batch.append(filename)


            if len(img_path_batch) != 5:
                continue
            else:
                try:
                    image_batch = torch.tensor(np.stack(
                        [preprocess(Image.open(io.BytesIO(open(filename, 'rb').read())).convert("RGB")) for filename in
                         img_path_batch]
                    )).to(device)
                    with torch.no_grad():
                        image_features = clip_model.encode_image(image_batch).cpu().detach().numpy()
                except PIL.UnidentifiedImageError:
                    print(f'Cannot identify image file {filename}')
                    continue

                d['visual_features'] = image_features.astype(np.float32)

                # Clean the description from html formats
                d['description'] = BeautifulSoup(d['description'], 'html.parser').get_text().replace('\n', ' ')

                # Extract the first sentence of the discription to avoid too long text
                nlp = spacy.load('en_core_web_sm')
                d['description'] = list(nlp(d['description']).sents)[0].text.replace('\n', ' ').strip() if len(d['description'].split()) > 128 else d['description']

                # {'original_text': 'Our landmark tree in town was about to be destroyed and cleared for a new mall. ', Done
                #  'album_id': '72157605930515606',
                #  'photo_flickr_id': '2627795780', Done
                #  'setting': 'first-2-pick-and-tell',
                #  'worker_id': 'SY6QQXJCXXMNCYP',
                #  'story_id': '30355', Done
                #  'tier': 'story-in-sequence',
                #  'worker_arranged_photo_order': 0, Done
                #  'text': 'our landmark tree in town was about to be destroyed and cleared for a new mall .',
                #  'storylet_id': '151775', Done
                #  'clip_embedding': 0}

                # {'img_ids': ['4541316341', '4541948966', '4541948308', '4541946480', '4541946036'],
                # 'sis_sents': ['The park today was a good choice, ', 'This guy is always playing the same song.', 'We saw some people in a boat and quickly became jealous.', "He said don't even think of getting on the ferris wheel.", 'So we, just went inside.'],
                # 'storylet_ids': ['206870', '206871', '206872', '206873', '206874'],
                # 'worker_arranged_photo_orders': [0, 1, 2, 3, 4],
                # 'story_id': '41374',
                # 'title': 'Sechseläuten 2010',
                # 'description': ''}

                for j in range(len(d['sis_sents'])):
                    item = {}
                    item['title'] = d['title']
                    item['description'] = d['description']
                    item['original_text'] = d['sis_sents'][j]
                    item['photo_flickr_id'] = d['img_ids'][j]
                    item['storylet_id'] = d['storylet_ids'][j]
                    item['worker_arranged_photo_order'] = d['worker_arranged_photo_orders'][j]
                    item['story_id'] = d['story_id']
                    item['clip_embedding'] = clip_embedding_index

                    tit_desc = d['title'] + ". " if d['description'].strip() == '' else d['title'] + ". " + d['description']
                    if j == 0:
                        item['context'] = tit_desc
                    elif j < context_num:
                        item['context'] = tit_desc + " ".join(d['sis_sents'][:j])
                    else:
                        item['context'] = " ".join(d['sis_sents'][j-context_num:j])


                    all_image_embeddings.append(torch.from_numpy(image_features[j]).unsqueeze(0))
                    with torch.no_grad():
                        all_context_embeddings.append(clip_model.encode_text(clip.tokenize(item['context'].strip(), context_length=77, truncate=True).to(device)).cpu().detach().float())
                    all_captions.append(item)
                    clip_embedding_index += 1

                    torch.cuda.empty_cache()
                    # print(item)
                    # print("==============================================")

                    # if (i + 1) % 10000 == 0:
                    #     with open(out_path, 'wb') as f:
                    #         pickle.dump({"clip_embedding": torch.cat(all_image_embeddings, dim=0),
                    #                      "context_embedding":torch.cat(all_context_embeddings, dim=0),
                    #                      "captions": all_captions},
                    #                     f)



        print(all_captions[1234])
        print(all_image_embeddings[1234])
        print(all_context_embeddings[1234])



        print(len(all_captions))
        print(len(all_image_embeddings))
        print(len(all_context_embeddings))
        print(clip_embedding_index)
        assert clip_embedding_index == len(all_captions)
        assert len(all_image_embeddings) == len(all_captions)
        assert len(all_context_embeddings) == len(all_captions)
        print("all image embeddings len:", len(all_image_embeddings))
        print("all context embeddings len:", len(all_context_embeddings))
        print(all_image_embeddings[-1].shape)
        print("all captions len:", len(all_captions))
        # torch.Size([1, 512])
        # all captions len: 15096

        with open(out_path, 'wb') as f:
            pickle.dump({"clip_embedding": torch.cat((all_image_embeddings), dim=0),
                         "context_embedding": torch.cat(all_context_embeddings, dim=0),
                         "captions": all_captions},
                        f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', type=str, default="RN50x4", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--annot_dir', type=str, default="/scratch/song0018/vist_data/annotations/", help='path to the annotation json files directory')
    parser.add_argument('--image_dir', type=str, default="/scratch/song0018/vist_data/images/", help='path to the image files directory')
    parser.add_argument('--context_num', type=int, default=1, choices=(1, 2, 3, 4, 5))

    args = parser.parse_args()
    exit(main(args.clip_model_type, args.annot_dir, args.image_dir, args.context_num))
