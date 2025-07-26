import os
import functools
import math

import torch
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, AdamW, AutoTokenizer, OPTModel
from tqdm import tqdm
import os
import pickle
import wandb
import sys
import argparse
import json
from typing import Tuple, Union
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


from model_context_text_gpt import *

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class ClipVistDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item][0]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item][0] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item][0] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        #mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask

        prev_tokens = self.captions_tokens[item][1]
        # print("===========================================")
        # print("prev_tokens 0: ", prev_tokens)
        # print(prev_tokens.shape)
        prev_padding = self.max_seq_len - prev_tokens.shape[0]
        if prev_padding > 0:
            prev_tokens = torch.cat((prev_tokens, torch.zeros(prev_padding, dtype=torch.int64) - 1))
            self.captions_tokens[item][1] = prev_tokens
        elif prev_padding < 0:
            prev_tokens = prev_tokens[:self.max_seq_len]
            self.captions_tokens[item][1] = prev_tokens
        # print("prev_tokens 1: ", prev_tokens)
        # print(prev_tokens.shape)
        prev_mask = prev_tokens.ge(0)  # mask is zero where we out of sequence
        # print("prev_mask: ", prev_mask)
        # print(prev_mask.shape)
        prev_tokens[~prev_mask] = 0
        # print("prev_tokens 2: ", prev_tokens)
        # print(prev_tokens.shape)


        # prev_mask = prev_mask.float()
        # prev_mask = torch.cat((torch.ones(self.prefix_length), prev_mask), dim=0)
        # adding prefix mask
        mask = torch.cat((torch.ones(self.prefix_length+1+self.max_seq_len+1+1), mask), dim=0)

        return tokens, mask, prev_tokens

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask, prev_tokens = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        # tokens, mask, prev_tokens, prefix
        return tokens, mask, prev_tokens, prefix

    def __init__(self, data_path: str,  prefix_length: int,
                 normalize_prefix: bool, lm: str, split: str):
        if lm == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        elif lm == 'opt':
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        else:
            raise Exception("Unknown language model type")

        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Caption data size is %0d" % len(all_data["clip_embedding"]))

        sys.exit(0)
        sys.stdout.flush()

        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]

        # {'original_text': 'Large tree with many outstretching branches and leaves.',
        # 'album_id': '72157605930515606',
        # 'photo_flickr_id': '2701863545',
        # 'photo_order_in_story': 0,
        # 'worker_id': 'BNCNALGFUMO75H3',
        # 'text': 'large tree with many outstretching branches and leaves.',
        # 'tier': 'descriptions-in-isolation', 'clip_embedding': 0}

        info = json.load(open(os.path.join("/hpc/shared/uu_vl/yingjin_datasets/vist_data/annotations/dii", f"{split}.description-in-isolation.json")))
        album_info = {album['id']: album for album in info['albums']}

        # create a dictionary of captions with story_id as key to avoid iterating over the list of captions later
        captions_dict = {}
        for item in captions_raw:
            album_id = item['album_id']
            if album_id not in captions_dict:
                captions_dict[album_id] = []
            captions_dict[album_id].append(item)

        for album_id, album in captions_dict.items():
            assert len(album) > 1
            for i in range(len(album)):
                caption = album[i]
                if caption['photo_order_in_story'] == 0:
                    caption['prev_sent'] = album_info[album_id]['title'].strip() + "."
                    caption['used_as_prev'] = False
                else:
                    caption['prev_sent'] = album[i - 1]['original_text']
                    if caption['photo_order_in_story'] == 4:
                        caption['used_as_prev'] = True
                    else:
                        caption['used_as_prev'] = False

        # len_available_sent = len(captions_raw)
        # unavailable_story_ids = []
        # for key in captions_dict.keys():
        #     if len(captions_dict[key]) != 5:
        #         unavailable_story_ids.append(key)
        #         len_available_sent -= len(captions_dict[key])

        context_captions_raw = []

        for caption in captions_raw:
            album_id = caption['album_id']
            caption['prev_sent'] = ""
            caption['original_text'] = caption['original_text'].strip()
            if caption['photo_order_in_story'] == 0:
                caption['prev_sent'] = album_info[album_id]['title'].strip() + "." + album_info[album_id]['description'].strip()
                #caption['prev_sent'] = album_info[album_id]['title'].strip() + ". "
            else:
                prev_order = caption['photo_order_in_story'] - 1
                album_items = captions_dict.get(album_id)
                for item in album_items:
                    if item['photo_order_in_story'] == prev_order and item['used_as_prev'] == False:
                        assert item['album_id'] == caption['album_id']
                        caption['prev_sent'] = item['original_text'].strip()
                        item['used_as_prev'] = True
                        break
            context_captions_raw.append(caption)
            # available_story_ids.append(story_id)


        # assert len(context_captions_raw) == len(captions_raw)
        # print(len(list(set(available_story_ids))))
        # print(len(captions_dict.keys()))
        # print(len(unavailable_story_ids))
        #assert len(list(set(available_story_ids))) == len(captions_dict.keys()) - len(unavailable_story_ids)

        self.image_ids = [caption["photo_flickr_id"] for caption in context_captions_raw]
        self.captions = [caption['original_text'] for caption in context_captions_raw]


        # if os.path.isfile(f"{data_path[:-4]}_tokens_context.pkl"):
        #     with open(f"{data_path[:-4]}_tokens_context.pkl", 'rb') as f:
        #         self.captions_tokens, self.caption2embedding, max_seq_len = pickle.load(f)
        # else:
        self.captions_tokens = []
        self.prev_sents_tokens = []
        self.caption2embedding = []
        max_seq_len = 0

        for caption in context_captions_raw:
                # if caption['worker_arranged_photo_order'] == 0:
                #     assert caption['prev_sent'] == " "
                # else:
            if not len(caption['prev_sent'].split()) > 0:
                print(caption)
            self.captions_tokens.append([torch.tensor(self.tokenizer.encode(caption['original_text'].strip()+ self.tokenizer.eos_token), dtype=torch.int64),
                                         torch.tensor(self.tokenizer.encode(caption['prev_sent'].strip()+ self.tokenizer.eos_token),dtype=torch.int64)])
            self.caption2embedding.append(caption["clip_embedding"])
            max_seq_len = max(max_seq_len, self.captions_tokens[-1][0].shape[0])

        assert len(self.captions_tokens) == len(self.caption2embedding)
        assert len(self.captions_tokens) == len(self.captions)
        assert len(self.captions_tokens) == len(self.image_ids)

        with open(f"{data_path[:-4]}_tokens_context.pkl", 'wb') as f:
            pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)

        all_len = torch.tensor([len(self.captions_tokens[i][0]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        all_len_prev = torch.tensor([len(self.captions_tokens[i][1]) for i in range(len(self))]).float()
        self.max_seq_len_prev = min(int(all_len_prev.mean() + all_len_prev.std() * 10), int(all_len_prev.max()))
        self.max_seq_len_prev = self.max_seq_len + 32
        print("Max sequence length is %0d" % self.max_seq_len)
        print("Max sequence length (prev sentence) is %0d" % self.max_seq_len_prev)

def train(train_dataset: ClipVistDataset, val_dataset: ClipVistDataset, model: ContextClipCaptionModel, args,
          output_dir: str = ".", output_prefix: str = ""):

    batch_size = args.bs
    epochs = args.epochs
    lr = args.learning_rate
    weight_decay = args.weight_decay
    warmup_ratio = args.warmup_ratio

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    device = torch.device('cuda:0')
    model = model.to(device)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    if args.linear_scheduler:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        #warmup_steps = epochs * len(train_dataloader) * warmup_ratio
        warmup_steps = 800
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader))
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, correct_bias=True)
        decay_steps = len(train_dataloader) * epochs
        warmup_steps = decay_steps * args.warmup_ratio
        warmup_steps = 800
        resample = args.resample
        def lr_schedule_fn(iteration, iter_per_epoch):
            if iteration < warmup_steps:
                lr_multiplier = iteration / (warmup_steps)
            else:
                lr_multiplier = 0.5 * \
                                (1. + math.cos(
                                    math.pi * (iteration - warmup_steps) / (epochs * iter_per_epoch - warmup_steps)))
            return lr_multiplier

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, functools.partial(lr_schedule_fn, iter_per_epoch=len(train_dataloader) * resample)
        )




    if args.run_wandb:
        wandb.init(project='CLIP_prefix_CAP-context',
               config={
                   "learning_rate": lr,
                   "epochs": epochs,
                   "batch_size": batch_size,
                   "warmup_steps": warmup_steps,
                   "training_steps": epochs * len(train_dataloader),
                   "weight_decay": weight_decay,
                   "resample": args.resample,
                   "mask_type": args.mask_type,
                   "warmup_ratio": warmup_ratio,
                   "prefix_length": args.prefix_length,
                   "prefix_length_clip": args.prefix_length_clip,
                   "is_resnet": args.is_rn,
                   "is_only_prefix": args.only_prefix,
                   "normalize_prefix": args.normalize_prefix,
                   "mapping_type": args.mapping_type,
                   "num_transformer_layers": args.num_layers,
                   "models_saved_dir": output_dir,
                   "train_data": args.train_data,
                   "use_pretrained": args.use_pretrained,
                   "fn_prefix": args.fn_prefix,
                   "linear_scheduler": args.linear_scheduler,
                   "text_first": args.text_first,
               })
        wandb.watch(model, log='all')

    loss_per_epoch_train = []
    loss_per_epoch_val = []


    for epoch in range(epochs):


        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        model.train()
        optimizer.zero_grad()
        accumulated_loss = 0.0


        for idx, (tokens, mask, prev_tokens, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prev_tokens, prefix = tokens.to(device), mask.to(device), prev_tokens.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, prev_tokens, args, mask)
            logits = outputs.logits[:, train_dataset.prefix_length - 1 + 1 + train_dataset.max_seq_len + 2: -1]
            # print("==================================")
            # print(tokens.shape)
            # print(outputs.logits.shape)
            # print(logits.shape)
            # sys.exit(0)
            # print("==================================")

            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_value = loss.item()
            progress.set_postfix({"loss": loss_value})
            progress.update()

            accumulated_loss += loss_value

            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{args.train_data[60:-4]}_{args.prefix_length}_{args.prefix_length_clip}_{output_prefix}_latest.pt"),
                )
        progress.close()
        loss_per_epoch_train.append(accumulated_loss / len(train_dataloader))
        print('train_loss_per_epoch: ', loss_per_epoch_train)

        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir,
                             f"{args.train_data[60:-4]}_{args.prefix_length}_{args.prefix_length_clip}_{output_prefix}-{epoch:03d}.pt"),
            )

        sys.stdout.flush()
        progress = tqdm(total=len(val_dataloader), desc=f"Evaluating...")
        model.eval()
        val_loss = 0.0

        torch.cuda.empty_cache()

        with torch.no_grad():
            torch.cuda.empty_cache()
            for idx, (tokens, mask, prev_tokens, prefix) in enumerate(val_dataloader):
                tokens, mask, prev_tokens, prefix = tokens.to(device), mask.to(device), prev_tokens.to(device), prefix.to(device, dtype=torch.float32)
                outputs = model(tokens, prefix, prev_tokens, args, mask)
                logits = outputs.logits[:, train_dataset.prefix_length - 1 + 1 + val_dataset.max_seq_len + 2: -1]
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

                val_loss += loss.item()
            progress.update()
        progress.close()
        model.train()

        loss_per_epoch_val.append(val_loss/len(val_dataloader))
        print('val_loss_per_epoch: ', loss_per_epoch_val)

        if args.run_wandb:
            wandb.log({
            'train_loss_final': loss_per_epoch_train[-1],
            'train_loss_lowest': min(loss_per_epoch_train),
            'train_loss_lowest_index': loss_per_epoch_train.index(min(loss_per_epoch_train)),
            'valid_loss_final': loss_per_epoch_val[-1],
            'valid_loss_lowest': min(loss_per_epoch_val),
            'valid_loss_lowest_index': loss_per_epoch_val.index(min(loss_per_epoch_val)),
        })


        with open(os.path.join(output_dir, f"loss_per_epoch.json"), 'w') as f:
            json.dump({'train': loss_per_epoch_train, 'val': loss_per_epoch_val}, f)
    return model





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/dii_RN50x4_train.pkl')
    parser.add_argument('--val_data', default='/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/dii_RN50x4_val.pkl')
    parser.add_argument('--out_dir', default='/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/checkpoints/')
    parser.add_argument('--fn_prefix', default='vist_prev_sent', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=20)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='transformer', choices=('mlp', 'transformer'))
    parser.add_argument('--lang_model', type=str, default='gpt2', choices=('gpt2', 'opt'))
    parser.add_argument('--mask_type', type=str, default='ones', choices=('ones', 'default'))
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--resample', type=int, default=3)
    parser.add_argument('--text_first', dest='text_first', action='store_true')
    parser.add_argument('--run_wandb', dest='run_wandb', action='store_true')
    parser.add_argument('--linear_scheduler', dest='linear_scheduler', action='store_true')
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--use_pretrained', dest='use_pretrained', action='store_true')
    parser.add_argument('--pretrained_weights', default='/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/pretrained/')

    args = parser.parse_args()
    prefix_length = args.prefix_length # length of input caption sequence

    train_dataset = ClipVistDataset(args.train_data, prefix_length, normalize_prefix=args.normalize_prefix, lm=args.lang_model, split='train')
    val_dataset = ClipVistDataset(args.val_data, prefix_length, normalize_prefix=args.normalize_prefix, lm=args.lang_model, split='val')
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]

    if args.only_prefix:
        model = ContextClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train only prefix (frozen LM)")
    else:
        model = ContextClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                 num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train both prefix and LM")


    if args.use_pretrained and os.path.isfile(args.pretrained_weights):
        print("Initialize with pretrained weights ...")
        model.load_state_dict(torch.load(args.pretrained_weights, map_location=torch.device('cpu')))
    else:
        print("Initialize with random weights ...")

    print('Number of trainable parameters is',
          sum(para.numel() for para in model.parameters() if para.requires_grad))

    train(train_dataset, val_dataset, model, args, output_dir=args.out_dir, output_prefix=args.fn_prefix)




if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     train_data = "/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/dii_RN50x4_train.pkl"
#     prefix_length = 40
#
#     train_dataset = ClipVistDataset(train_data, prefix_length, normalize_prefix=True)
#     print(len(train_dataset[0]))