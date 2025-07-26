import os
import json
import numpy as np
import argparse
import clip
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.LOAD_TRUNCATED_IMAGES = True
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
import collections
import os
import pathlib
import json
import generation_eval_utils
import pprint
import warnings
from packaging import version
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')




class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image':image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            if device == 'cuda:0':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device)

    #as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

    per = w*np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates


def get_refonlyclipscore(model, references, candidates, device):
    '''
    The text only side for refclipscore
    '''
    if isinstance(candidates, list):
        candidates = extract_all_captions(candidates, model, device)

    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    flattened_refs = extract_all_captions(flattened_refs, model, device)

    if version.parse(np.__version__) < version.parse('1.21'):
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
        flattened_refs = sklearn.preprocessing.normalize(flattened_refs, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')

        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
        flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs**2, axis=1, keepdims=True))

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, cand in tqdm.tqdm(enumerate(candidates)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())
        per.append(np.max(all_sims))

    return np.mean(per), per


def main():
    compute_other_ref_metrics = True
    save_per_instance = False
    use_references = True
    compute_reco = True

    image_dir = '/scratch/song0018/vist_data/images/test_images'
    data_dir = '/scratch/song0018/vist_data/results/'

    candidates_json = "data/gt.json"
    gt_json = "data/gt.json"
    reco_json = "data/dedup_reco_preds.json"



    # image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
    #                if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    # image_ids = [pathlib.Path(path).stem for path in image_paths]


    with open(candidates_json) as f:
        candidates_all = json.load(f)

    with open(gt_json) as f:
        gt_all = json.load(f)

    with open(reco_json) as f:
        reco_all = json.load(f)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')
    model, transform = clip.load("ViT-B/32", device=device, jit=False)  #RN101
    model.eval()

    image_ids = []
    candidates = []
    references = []
    non_count = 0
    if not compute_reco:
        for key, pred_story in candidates_all.items():
            # image_ids.extend(key.split('_')[1:])
            pred_sents = sent_tokenize(pred_story)
            gt_sents = sent_tokenize(gt_all[key])
            if len(pred_sents) < 5 or len(gt_sents) < 5:
                non_count += 1
                continue
            references.extend(gt_sents[:5])
            candidates.extend(pred_sents[:5])
            image_ids.extend(key.split('_')[1:])

    else:
        for item in reco_all:
            album_id, pred_story_str = item['album_id'], item['pred_story_str']
            pred_sents = sent_tokenize(pred_story_str)
            for key, pred_story in candidates_all.items():
                if album_id == key.split('_')[0]:
                    gt_id = key
                    break
            gt_sents = sent_tokenize(gt_all[gt_id])
            if len(pred_sents) < 5 or len(gt_sents) < 5:
                non_count += 1
                continue
            references.extend(gt_sents[:5])
            candidates.extend(pred_sents[:5])
            image_ids.extend(gt_id.split('_')[1:])

    print(non_count)
    assert len(image_ids) == len(candidates) == len(references)
    print(len(image_ids))

    img_paths = [os.path.join(image_dir, image_id + '.jpg') for image_id in image_ids]

    image_paths = []
    for img_path in img_paths:
        if os.path.exists(img_path):
            image_paths.append(img_path)
        else:
            image_paths.append(img_path.replace('.jpg', '.png'))



    image_feats = extract_all_images(
        image_paths, model, device, batch_size=1024, num_workers=8)

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, candidates, device)

    if use_references:
        # get text-text clipscore
        _, per_instance_text_text = get_refonlyclipscore(
            model, references, candidate_feats, device)
        # F-score
        refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)
        scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
                  for image_id, clipscore, refclipscore in
                  zip(image_ids, per_instance_image_text, refclipscores)}

    else:
        scores = {image_id: {'CLIPScore': float(clipscore)}
                  for image_id, clipscore in
                  zip(image_ids, per_instance_image_text)}
        print('CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()])))


    print(f'======================================== {candidates_json.upper()} =============================================')

    if use_references:
        if compute_other_ref_metrics:
            other_metrics = generation_eval_utils.get_all_metrics(references, candidates)
            for k, v in other_metrics.items():
                if k == 'bleu':
                    for bidx, sc in enumerate(v):
                        print('BLEU-{}: {:.4f}'.format(bidx+1, sc))
                else:
                    print('{}: {:.4f}'.format(k.upper(), v))
        print('CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()])))
        print('RefCLIPScore: {:.4f}'.format(np.mean([s['RefCLIPScore'] for s in scores.values()])))



if __name__ == '__main__':
    main()





