import os
from model_new_concat import *

import functools
import math
import clip
import torch
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AdamW, AutoTokenizer
from tqdm import tqdm
import pickle
import wandb
import sys
import argparse
import json
from typing import Tuple, Union
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from info_nce import InfoNCE

import spacy
from bs4 import BeautifulSoup




class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class ClipVistDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        # print(tokens)
        # print(mask)
        # # tensor([ 4342,   318, 24980,   290,   262,  1641,    13,   679,   750,   340,
        # #            13,   775,   389,   523,  6613,    13, 50256,     0,     0,     0,
        # #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        # #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        # #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        # #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        # #             0,     0,     0])
        # # tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        # #         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        # #         1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        # #         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        # #         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]].cuda()

        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)

        prev_tokens = self.contexts[self.context2embedding[item]].cuda()

        # normalize prev_tokens
        prev_tokens = prev_tokens.float()
        prev_tokens /= prev_tokens.norm(dim=-1, keepdim=True)
        prev_tokens = prev_tokens.squeeze()
        # prev_tokens = prev_tokens.float()
        # prev_tokens = prev_tokens / prev_tokens.norm(2, -1)
        # prev_tokens = prev_tokens.squeeze()



        if self.text_first:
            concatenated_prefix = torch.cat((prev_tokens, prefix), dim=0)
        else:
            concatenated_prefix = torch.cat((prefix, prev_tokens), dim=0)



        return tokens, mask, concatenated_prefix, prefix

    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False, text_first = True):
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        # self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf',
        #                                                use_fast=False,
        #                                                use_auth_token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.text_first = text_first
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Caption data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()

        self.prefixes = all_data["clip_embedding"]
        self.contexts = all_data['context_embedding']
        captions_raw = all_data["captions"]

        # string_list = []
        # for i in range(len(captions_raw)):
        #     string_list.append(captions_raw[i]['context'])
        # import nltk
        # nltk.download('punkt')
        # tokenized_list = [nltk.word_tokenize(sentence) for sentence in string_list]
        # token_lengths = [len(sentence_tokens) for sentence_tokens in tokenized_list]
        # mean_token_length = sum(token_lengths) / len(token_lengths)
        # print(max(set(token_lengths), key=token_lengths.count))
        # print("Mean token length is %0d" % mean_token_length)
        # print("Max token length is %0d" % max(token_lengths))
        # print("Min token length is %0d" % min(token_lengths))
        # assert len(tokenized_list) == len(captions_raw)
        # print(len(string_list))
        # sys.exit(0)


        self.image_ids = [caption["photo_flickr_id"] for caption in captions_raw]
        self.captions = [caption['original_text'] for caption in captions_raw]


        self.captions_tokens = []
        self.caption2embedding = []
        self.context2embedding = []
        max_seq_len = 0
        for caption in captions_raw:
            self.captions_tokens.append(
                torch.tensor(self.tokenizer.encode(caption['original_text'].strip() + self.tokenizer.eos_token),
                             dtype=torch.int64))
            self.caption2embedding.append(caption["clip_embedding"])
            self.context2embedding.append(caption["clip_embedding"])
            max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
        with open(f"{data_path[:-4]}_tokens_prevMN.pkl", 'wb') as f:
            pickle.dump([self.captions_tokens, self.caption2embedding, self.context2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        print("Max sequence length is %0d" % self.max_seq_len)

def train(train_dataset: ClipVistDataset, val_dataset: ClipVistDataset, model: ClipCaptionModel, args,
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


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    if args.linear_scheduler:
        #https://github.com/pytorch/pytorch/issues/40497#issuecomment-1047923282
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-05)
        warmup_steps = epochs * len(train_dataloader) * warmup_ratio
        #warmup_steps = 1333
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader))
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, correct_bias=True)
        decay_steps = len(train_dataloader) * epochs
        warmup_steps = decay_steps * args.warmup_ratio
        warmup_steps = 1333
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

    # def _get_reference_text_image_features(labels, image_features):
    #     # last_padding_index = torch.where(labels == 0)[1]
    #     # first_non_padding_index = last_padding_index[0]
    #     # trimmed_labels = labels[:, :first_non_padding_index]
    #     # print(labels)
    #     # print(last_padding_index)
    #     # print(first_non_padding_index)
    #     ref_text = train_dataset.tokenizer.batch_decode(sequences=labels, skip_special_tokens=True)
    #     clean_ref_text = [string.replace("!", "") for string in ref_text]
    #
    #     if args.train_data.find('RN50x4') > -1:
    #         clip_model_name = 'RN50x4'
    #     elif args.train_data.find('ViT') > -1:
    #         clip_model_name = 'ViT-B/32'
    #     else:
    #         raise ValueError('Unknown clip model')
    #     clip_model, _ = clip.load(clip_model_name, device=device)
    #     clip_model.eval()
    #     clip_model.to(device)
    #
    #     ref_text_inputs = clip.tokenize(clean_ref_text, truncate=True).to(device)
    #     text_encoded = clip_model.encode_text(ref_text_inputs)
    #     ref_text_features = text_encoded / text_encoded.norm(dim=-1, keepdim=True)
    #
    #     if len(image_features.size()) == 3:
    #         image_features = image_features[:, 0, :]
    #
    #     # ref_img_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     # ref_img_features = ref_img_features.half()
    #     ref_img_features = image_features.half()
    #
    #     return clean_ref_text, ref_text_features, ref_img_features

    def _get_reference_text_image_features(labels, image_features):
        last_padding_index = torch.where(labels == 0)[1]
        first_non_padding_index = last_padding_index[0]
        trimmed_labels = labels[:, :first_non_padding_index]
        ref_text = train_dataset.tokenizer.batch_decode(sequences=trimmed_labels, skip_special_tokens=True)

        if args.train_data.find('RN50x4') > -1:
            clip_model_name = 'RN50x4'
        elif args.train_data.find('ViT') > -1:
            clip_model_name = 'ViT-B/32'
        else:
            raise ValueError('Unknown clip model')
        clip_model, _ = clip.load(clip_model_name, device=device)
        clip_model.eval()
        clip_model.to(device)

        ref_text_inputs = clip.tokenize(ref_text, truncate=True).to(device)
        text_encoded = clip_model.encode_text(ref_text_inputs)
        ref_text_features = text_encoded / text_encoded.norm(dim=-1, keepdim=True)

        # if len(image_features.size()) == 3:
        #     image_features = image_features[:, 0, :]
        #
        # ref_img_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # ref_img_features = ref_img_features.half()
        ref_img_features = image_features.half()
        return ref_text, ref_text_features, ref_img_features


    def _normalize(feature):
        norm = feature.norm(p=2, dim=1, keepdim=True)
        feature = feature.div(norm + 1e-16)
        return feature


    if args.run_wandb:
        wandb.init(project='CLIP_prefix_CAP-contrastive',
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
                   "lambda_text": args.lambda_text,
                   "lambda_contrastive_loss": args.lambda_contrastive_loss,
                   "start_contrastive_at_epoch": args.start_contrastive_at_epoch,
               })
        wandb.watch(model, log='gradients')

    loss_per_epoch_train = []
    loss_lm_per_epoch_train = []
    loss_contras_per_epoch_train = []
    loss_per_epoch_val = []
    loss_lm_per_epoch_val = []
    loss_contras_per_epoch_val = []

    # before training
    # https://github.com/pytorch/pytorch/issues/40497#issuecomment-656312357



    for epoch in range(epochs):

        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        model.train()
        optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_loss_lm = 0.0
        accumulated_loss_contra = 0.0


        for idx, (tokens, mask, prefix, image_features) in enumerate(train_dataloader):
            # This prefix is after concatenating the previous sentence and the image prefix
            model.zero_grad()
            # print("============tokens==============")
            # print(tokens.shape)
            # print("============mask==============")
            # print(mask.shape)
            # print("============prefix==============")
            # print(prefix.shape)
            # print("============image_features==============")
            # print(image_features.shape)


            tokens, mask, prefix, image_features = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32), image_features.to(device, dtype=torch.float32)

            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]       # train_dataset.prefix_length = 20

            # print("==================================")
            # print(tokens.shape)
            # print(outputs.logits.shape)
            # print(logits.shape)
            # break
            # torch.Size([32, 62])
            # torch.Size([32, 102, 50257]) 40+62 = 102
            # torch.Size([32, 62, 50257])

            lm_loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            accumulated_loss_lm += lm_loss.item()

            if args.run_wandb:
                wandb.log({'lm_loss_train': lm_loss})

            if epoch < args.start_contrastive_at_epoch or args.start_contrastive_at_epoch == -1:
                loss = lm_loss
            else:
                contrastive_loss = InfoNCE()

                hyp_text = train_dataset.tokenizer.batch_decode(
                    sequences=torch.argmax(logits, -1), skip_special_tokens=True)

                decoder_hidden_states = outputs.hidden_states[0][:, train_dataset.prefix_length - 1:-1]
                projected_pred_feat = _normalize(model.projection_layer(decoder_hidden_states.mean(dim=1)))
                outputs.projected_pred_feat = projected_pred_feat.half()
                hyp_text_features = outputs.projected_pred_feat

                ref_text, ref_text_features, ref_img_features = _get_reference_text_image_features(
                    labels=tokens, image_features=image_features)


                # Contrastive loss
                # (1) generated text (outputs of mapping network) vs. generated image
                # (2) generated text (outputs of mapping network) vs. GT text
                text_info_nce = contrastive_loss(
                    query=hyp_text_features, positive_key=ref_text_features)
                text_img_info_nce = contrastive_loss(
                    query=hyp_text_features, positive_key=ref_img_features)

                combined_info_nce = args.lambda_text * text_info_nce + \
                                    (1 - args.lambda_text) * text_img_info_nce

                if args.run_wandb:
                    wandb.log({'contra_loss_train': combined_info_nce})

                loss = lm_loss + args.lambda_contrastive_loss * combined_info_nce
                accumulated_loss_contra += combined_info_nce.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_value = loss.item()
            progress.set_postfix({"loss": loss_value})
            progress.update()

            accumulated_loss += loss_value

            if args.run_gemini:
                header = args.train_data[40:-4]
            else:
                header = args.train_data[60:-4]

            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{header}_{args.prefix_length}_{args.prefix_length_clip}_{output_prefix}_latest.pt"),
                )
        progress.close()
        loss_per_epoch_train.append(accumulated_loss / len(train_dataloader))
        loss_lm_per_epoch_train.append(accumulated_loss_lm / len(train_dataloader))
        loss_contras_per_epoch_train.append(accumulated_loss_contra / len(train_dataloader))

        print('train_loss_per_epoch: ', loss_per_epoch_train)
        print('train_loss_lm_per_epoch: ', loss_lm_per_epoch_train)
        print('train_loss_contras_per_epoch: ', loss_contras_per_epoch_train)

        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir,
                             f"{header}_{args.prefix_length}_{args.prefix_length_clip}_{output_prefix}-{epoch:03d}.pt"),
            )


        ########################### Validation #################################

        sys.stdout.flush()
        progress = tqdm(total=len(val_dataloader), desc=f"Evaluating...")
        model.eval()
        val_loss = 0.0
        val_loss_lm = 0.0
        val_loss_contra = 0.0

        with torch.no_grad():
            torch.cuda.empty_cache()
            for idx, (tokens, mask, prefix, image_features) in enumerate(val_dataloader):
                tokens, mask, prefix, image_features = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32), image_features.to(device, dtype=torch.float32)
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]
                lm_loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

                val_loss_lm += lm_loss.item()

                if args.run_wandb:
                    wandb.log({'lm_loss_val': lm_loss})

                if epoch < args.start_contrastive_at_epoch or args.start_contrastive_at_epoch == -1:
                    loss = lm_loss
                else:
                    contrastive_loss = InfoNCE()

                    hyp_text = train_dataset.tokenizer.batch_decode(
                        sequences=torch.argmax(logits, -1), skip_special_tokens=True)

                    decoder_hidden_states = outputs.hidden_states[0][:, train_dataset.prefix_length - 1:-1]
                    projected_pred_feat = _normalize(model.projection_layer(decoder_hidden_states.mean(dim=1)))
                    outputs.projected_pred_feat = projected_pred_feat.half()
                    hyp_text_features = outputs.projected_pred_feat

                    ref_text, ref_text_features, ref_img_features = _get_reference_text_image_features(
                        labels=tokens, image_features=image_features)
                    text_info_nce = contrastive_loss(
                        query=hyp_text_features, positive_key=ref_text_features)
                    text_img_info_nce = contrastive_loss(
                        query=hyp_text_features, positive_key=ref_img_features)

                    combined_info_nce = args.lambda_text * text_info_nce + \
                                        (1 - args.lambda_text) * text_img_info_nce

                    if args.run_wandb:
                        wandb.log({'contra_loss_val': combined_info_nce})

                    val_loss_contra += combined_info_nce.item()

                    loss = lm_loss + args.lambda_contrastive_loss * combined_info_nce

                val_loss += loss.item()
            progress.update()
        progress.close()
        model.train()

        loss_per_epoch_val.append(val_loss/len(val_dataloader))
        loss_lm_per_epoch_val.append(val_loss_lm/len(val_dataloader))
        loss_contras_per_epoch_val.append(val_loss_contra/len(val_dataloader))
        print('val_loss_per_epoch: ', loss_per_epoch_val)
        print('val_loss_lm_per_epoch: ', loss_lm_per_epoch_val)
        print('val_loss_contras_per_epoch: ', loss_contras_per_epoch_val)

        if args.run_wandb:
            wandb.log({
                'train_loss_final': loss_per_epoch_train[-1],
                'train_loss_lowest': min(loss_per_epoch_train),
                #'train_loss_lowest_index': loss_per_epoch_train.index(min(loss_per_epoch_train)),
                'valid_loss_final': loss_per_epoch_val[-1],
                'valid_loss_lowest': min(loss_per_epoch_val),
                #'valid_loss_lowest_index': loss_per_epoch_val.index(min(loss_per_epoch_val)),
            })


        with open(os.path.join(output_dir, f"loss_per_epoch.json"), 'w') as f:
            json.dump({'train': loss_per_epoch_train, 'val': loss_per_epoch_val}, f)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/sis_RN50x4_train.pkl')
    parser.add_argument('--val_data', default='/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/sis_RN50x4_val.pkl')
    parser.add_argument('--clip_model_type', type=str, default='RN50x4',choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--out_dir', default='/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/checkpoints/')
    parser.add_argument('--fn_prefix', default='vist_prev_sent', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=20)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='transformer', choices=('mlp', 'transformer'))
    parser.add_argument('--lang_model', type=str, default='gpt2', choices=('gpt2', 'gpt2-medium', 'gpt2-xl', 'opt'))
    parser.add_argument('--mask_type', type=str, default='ones', choices=('ones', 'default'))
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=2e-05)
    parser.add_argument('--weight_decay', type=float, default=1e-04)
    parser.add_argument('--warmup_ratio', type=float, default=0.2)
    parser.add_argument('--resample', type=int, default=3)
    parser.add_argument('--text_first', dest='text_first', action='store_true')
    parser.add_argument('--run_wandb', dest='run_wandb', action='store_true')
    parser.add_argument('--linear_scheduler', dest='linear_scheduler', action='store_true')
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--use_pretrained', dest='use_pretrained', action='store_true')
    parser.add_argument('--pretrained_weights', default='/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/pretrained/')
    parser.add_argument('--lambda_contrastive_loss', type=float, default=0.3)
    parser.add_argument('--lambda_text', type=float, default=0.0)
    parser.add_argument('--start_contrastive_at_epoch', type=int, default=6)
    parser.add_argument('--run_gemini', dest='run_gemini', action='store_true')


    args = parser.parse_args()
    prefix_length = args.prefix_length # length of input caption sequence


    train_dataset = ClipVistDataset(args.train_data, prefix_length, normalize_prefix=args.normalize_prefix, text_first=args.text_first)
    val_dataset = ClipVistDataset(args.val_data, prefix_length, normalize_prefix=args.normalize_prefix, text_first=args.text_first)

    if args.train_data.find('RN50') != -1:
        args.is_rn = True
    else:
        args.is_rn = False
    prefix_dim = 640*2 if args.is_rn else 512*2
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]

    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train only prefix (frozen LM)")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
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