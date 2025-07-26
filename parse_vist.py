import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str, data_dir:str, annotation_setting:str, split:str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/{annotation_setting}_{clip_model_name}_{split}.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    if annotation_setting == "dii":
        json_fn = ".description-in-isolation.json"
    elif annotation_setting == "sis":
        json_fn = ".story-in-sequence.json"
    else:
        raise Exception("Annotation json file is not found.")

    # annotation_file = os.path.join(data_dir, annotation_setting, split, json_fn)
    # if split == "test":
    #     info = json.load(open(os.path.join(data_dir, annotation_setting, split + json_fn)))
    # else:
    #     info_train = json.load(open(os.path.join(data_dir, annotation_setting, "train" + json_fn)))
    #     info_val = json.load(open(os.path.join(data_dir, annotation_setting, "val" + json_fn)))
    #     assert info_train.keys() == info_val.keys()
    #     info = {key: info_train[key] + info_val[key] for key in info_train}

    info = json.load(open(os.path.join(data_dir, annotation_setting, split + json_fn)))


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
        for album in info['albums']:
            album['split'] = split
        sis['albums'] += info['albums']
        sis['images'] += info['images']
        sis['annotations'] += info['annotations']

        data = sis['annotations']
        print("%0d captions (individual SIS) loaded from json " % len(data))


    available_data = []
    for i in tqdm(range(len(data))):
        d = data[i][0]
        img_id = d['photo_flickr_id']
        #filename = f"/hpc/shared/uu_vl/yingjin_datasets/vist_data/images/all_images/{int(img_id)}.jpg"
        filename_train = f"/hpc/shared/uu_vl/yingjin_datasets/vist_data/images/resized_images/train/{int(img_id)}.jpg"
        filename_test = f"/hpc/shared/uu_vl/yingjin_datasets/vist_data/images/resized_images/test/{int(img_id)}.jpg"
        if os.path.isfile(filename_train) or os.path.isfile(filename_test):
            available_data.append(d)

    print("%0d captions with available images loaded from json " % len(available_data))

    all_embeddings = []
    all_captions = []

    for i in tqdm(range(len(available_data))):
        d = available_data[i]
        img_id = d['photo_flickr_id']
        if split == "test":
            filename = f"/hpc/shared/uu_vl/yingjin_datasets/vist_data/images/resized_images/test/{int(img_id)}.jpg"
        else:
            filename = f"/hpc/shared/uu_vl/yingjin_datasets/vist_data/images/resized_images/train/{int(img_id)}.jpg"
        if not os.path.isfile(filename):
            continue
        else:
            image = io.imread(filename)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            assert i < len(available_data)

            all_embeddings.append(prefix)
            all_captions.append(d)
            if (i + 1) % 10000 == 0:
                with open(out_path, 'wb') as f:
                    pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)


    assert len(all_embeddings) == len(all_captions)
    print("all embeddings len:", len(all_embeddings))
    print(all_embeddings[-1].shape)
    print("all captions len:", len(all_captions))
    # torch.Size([1, 512])
    # all captions len: 15096

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', type=str, default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--data_dir', type=str, default="/hpc/shared/uu_vl/yingjin_datasets/vist_data/annotations/", help='path to the annotation json files directory')
    parser.add_argument('--annot_setting', type=str, default="dii",  choices=('dii', 'sis'))
    parser.add_argument('--data_split', type=str, default="test", choices=('train', 'val', 'test'))
    args = parser.parse_args()
    exit(main(args.clip_model_type, args. data_dir, args.annot_setting, args.data_split))
