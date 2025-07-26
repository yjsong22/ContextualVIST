import os
from model_context_text_gpt import ContextClipCaptionModel, ContextClipCaptionPrefix

import functools
import math

import torch
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AdamW, LlamaTokenizer, AutoTokenizer
from tqdm import tqdm
import pickle
import wandb
import sys
import argparse
import json
from typing import Tuple, Union
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from info_nce import InfoNCE
import clip



# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)


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
        # sys.exit(0)
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
        prev_mask = prev_mask.float()
        # print("prev_tokens 2: ", prev_tokens)
        # print(prev_tokens.shape)
        #  # prev_context = torch.cat([bos_token, context_text, eos_token], dim=1)


        prev_mask = torch.cat((torch.ones(1), prev_mask, torch.ones(1)), dim=0)


        # prev_mask = prev_mask.float()
        # prev_mask = torch.cat((torch.ones(self.prefix_length), prev_mask), dim=0)
        # adding prefix mask
        # print("===================== mask ======================")
        # print(mask)
        # print("===================== prev_mask ======================")
        # print(prev_mask)

        # prev_context = torch.cat([bos_token, context_text, eos_token], dim=1)
        # if args.text_first:
        #     concat_context = torch.cat([prev_context, prefix_projections, bos_token_generation, embedding_text], dim=1)
        # else:
        #     concat_context = torch.cat([prefix_projections, prev_context, bos_token_generation, embedding_text], dim=1)

        if self.text_first:
            # concat_mask = torch.cat((prev_mask, torch.ones(self.prefix_length+1)), dim=0)
            # mask = torch.cat((concat_mask, mask), dim=0)
            mask = torch.cat((prev_mask, torch.ones(self.prefix_length+1), mask), dim=0)
        else:
            # concat_mask = torch.cat((torch.ones(self.prefix_length), prev_mask), dim=0)
            # concat_mask = torch.cat((concat_mask, torch.ones(1)), dim=0)
            # mask = torch.cat((concat_mask, mask), dim=0)
            mask = torch.cat((torch.ones(self.prefix_length), prev_mask, torch.ones(1), mask), dim=0)

        # # print(self.text_first)
        # # print("===================== tokens ======================")
        # # print(tokens)
        # # print("===================== mask_all ======================")
        # print(mask.size())
        # # print("===================== prev_tokens ======================")
        # # print(prev_tokens)
        # sys.exit()

        return tokens, mask, prev_tokens

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask, prev_tokens = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]

        # print(prefix.size())
        # torch.Size([640])

        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        # tokens, mask, prev_tokens, prefix
        return tokens, mask, prev_tokens, prefix

    def __init__(self, data_path: str,  prefix_length: int,
                 normalize_prefix: bool, text_first: bool):


        #self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=False, token=hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        self.tokenizer.pad_token = self.tokenizer.eos_token # https://github.com/arielnlee/Platypus/issues/14#issuecomment-1687605792
        self.tokenizer.padding_side = "right"
        #self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.text_first = text_first
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Caption data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()

        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]


        self.captions_tokens = []
        self.caption2embedding = []
        max_seq_len = 0

        for caption in captions_raw:

            # self.captions_tokens.append([torch.tensor(self.tokenizer.encode(caption['original_text'].strip()+self.tokenizer.eos_token), dtype=torch.int64),
            #                              torch.tensor(self.tokenizer.encode(caption['prev_sent'].strip()+self.tokenizer.eos_token),dtype=torch.int64)])
            self.captions_tokens.append([torch.tensor(self.tokenizer.encode(caption['original_text'].strip()+self.tokenizer.eos_token), dtype=torch.int64),
                                         torch.tensor(self.tokenizer.encode(caption['context'].strip()+self.tokenizer.eos_token),dtype=torch.int64)])
            self.caption2embedding.append(caption["clip_embedding"])
            max_seq_len = max(max_seq_len, self.captions_tokens[-1][0].shape[0])
        assert len(self.captions_tokens) == len(self.caption2embedding)
        # assert len(self.captions_tokens) == len(self.captions)
        # assert len(self.captions_tokens) == len(self.image_ids)
        # assert len(self.captions_tokens) == len_available_sent

        with open(f"{data_path[:-4]}_tokens_context.pkl", 'wb') as f:
            pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i][0]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        all_len_prev = torch.tensor([len(self.captions_tokens[i][1]) for i in range(len(self))]).float()
        self.max_seq_len_prev = min(int(all_len_prev.mean() + all_len_prev.std() * 10), int(all_len_prev.max()))

        print("Max sequence length is %0d" % self.max_seq_len)
        print("Max sequence length (context) is %0d" % self.max_seq_len_prev)

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
        #https://github.com/pytorch/pytorch/issues/40497#issuecomment-1047923282
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-04)
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

    def _get_reference_text_image_features_old(labels, image_features):
        last_padding_index = torch.where(labels == 0)[1]
        first_non_padding_index = last_padding_index[0]
        trimmed_labels = labels[:, :first_non_padding_index]
        ref_text = train_dataset.tokenizer.batch_decode(sequences=trimmed_labels, skip_special_tokens=True)
        print(ref_text)

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

    def _get_reference_text_image_features(labels, image_features, preds):
        # last_padding_index = torch.where(labels == 0)[1]
        # first_non_padding_index = last_padding_index[0]
        # trimmed_labels = labels[:, :first_non_padding_index]
        # print(labels)
        # print(last_padding_index)
        # print(first_non_padding_index)

        ref_text = train_dataset.tokenizer.batch_decode(sequences=labels, skip_special_tokens=True)
        clean_ref_text = [string.replace("!", "") for string in ref_text]

        if args.train_data.find('RN50x4') > -1:
            clip_model_name = 'RN50x4'
        elif args.train_data.find('ViT') > -1:
            clip_model_name = 'ViT-B/32'
        else:
            raise ValueError('Unknown clip model')
        clip_model, _ = clip.load(clip_model_name, device=device)
        clip_model.eval()
        clip_model.to(device)

        ref_text_inputs = clip.tokenize(clean_ref_text, truncate=True).to(device)
        text_encoded = clip_model.encode_text(ref_text_inputs)
        ref_text_features = text_encoded / text_encoded.norm(dim=-1, keepdim=True)

        hyp_text_inputs = clip.tokenize(preds, truncate=True).to(device)
        hyp_text_encoded = clip_model.encode_text(hyp_text_inputs)
        hyp_text_features = hyp_text_encoded / hyp_text_encoded.norm(dim=-1, keepdim=True)

        # if len(image_features.size()) == 3:
        #     image_features = image_features[:, 0, :]

        # ref_img_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # ref_img_features = ref_img_features.half()
        ref_img_features = image_features.half()

        return clean_ref_text, ref_text_features, ref_img_features, hyp_text_features


    def _normalize(feature):
        norm = feature.norm(p=2, dim=1, keepdim=True)
        feature = feature.div(norm + 1e-16)
        return feature

    if args.start_contrastive_at_epoch == -1:
        project_name = 'CLIP_prefix_CAP-context'
    else:
        project_name = 'CLIP_prefix_CAP-contrastive'


    if args.run_wandb:
        wandb.init(project=project_name,
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
                   "language_model": "gpt2-xl",
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

    for epoch in range(epochs):

        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        model.train()
        optimizer.zero_grad()

        accumulated_loss = 0.0
        accumulated_loss_lm = 0.0
        accumulated_loss_contra = 0.0

        for idx, (tokens, mask, prev_tokens, prefix) in tqdm(enumerate(train_dataloader)):
            model.zero_grad()
            tokens, mask, prev_tokens, prefix = tokens.to(device), mask.to(device), prev_tokens.to(device), prefix.to(device, dtype=torch.float32)
            # print("==================================")
            # print('tokens:',tokens.shape)
            # print('mask:',mask.shape)
            # print('prev_tokens:',prev_tokens.shape)
            # print(prefix.shape)
            # sys.exit(0)
            # tokens: torch.Size([50, 63])
            # mask: torch.Size([50, 149])
            # prev_tokens: torch.Size([50, 63])
            # torch.Size([50, 640])

            outputs = model(tokens, prefix, prev_tokens, args, mask)
            logits = outputs.logits[:, train_dataset.prefix_length - 1 + 1 + train_dataset.max_seq_len + 2: -1]

            # print("==================================")
            # print(tokens.shape) # torch.Size([50, 63])
            # print(outputs.logits.shape) # torch.Size([50, 149, 50257])
            # print(logits.shape) # torch.Size([50, 63, 50257])
            # # sys.exit(0)


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


                decoder_hidden_states = outputs.hidden_states[0][:, train_dataset.prefix_length - 1 + 1 + train_dataset.max_seq_len + 2: -1]
                projected_pred_feat = _normalize(model.projection_layer(decoder_hidden_states.mean(dim=1)))
                outputs.projected_pred_feat = projected_pred_feat.half()
                hyp_text_features = outputs.projected_pred_feat

                # print(logits.shape)
                # print(logits)
                # print('---------------------------------------------')
                # print(decoder_hidden_states.shape)
                # print(decoder_hidden_states)
                # sys.exit(0)


                # ref_text, ref_text_features, ref_img_features, hyp_text_features = _get_reference_text_image_features(
                #     labels=tokens, image_features=prefix, preds=hyp_text)

                ref_text, ref_text_features, ref_img_features = _get_reference_text_image_features_old(
                    labels=tokens, image_features=prefix)


                # print(outputs.hidden_states[0].shape)  # torch.Size([50, 149, 1280])
                # print(decoder_hidden_states.shape) # torch.Size([50, 63, 1280])
                # print(projected_pred_feat.shape) # torch.Size([50, 640])
                # print(outputs.projected_pred_feat.shape)

                if torch.isnan(hyp_text_features).any():
                    print('hyp_text_features is nan:', idx)
                if torch.isnan(ref_text_features).any():
                    print('ref_text_features is nan:', idx)
                if torch.isnan(ref_img_features).any():
                    print('ref_img_features is nan:', idx)


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
                    os.path.join(output_dir,
                                 f"{header}_{args.prefix_length}_{args.prefix_length_clip}_{output_prefix}_latest.pt"),
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

            for idx, (tokens, mask, prev_tokens, prefix) in tqdm(enumerate(val_dataloader)):
                tokens, mask, prev_tokens, prefix = tokens.to(device), mask.to(device), prev_tokens.to(device), prefix.to(device, dtype=torch.float32)

                outputs = model(tokens, prefix, prev_tokens, args, mask)
                logits = outputs.logits[:, val_dataset.prefix_length - 1 + 1 + val_dataset.max_seq_len + 2: -1]

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
                    #
                    decoder_hidden_states = outputs.hidden_states[0][:, val_dataset.prefix_length - 1 + 1 + val_dataset.max_seq_len + 2: -1]
                    projected_pred_feat = _normalize(model.projection_layer(decoder_hidden_states.mean(dim=1)))
                    outputs.projected_pred_feat = projected_pred_feat.half()
                    hyp_text_features = outputs.projected_pred_feat
                    ref_text, ref_text_features, ref_img_features = _get_reference_text_image_features_old(
                        labels=tokens, image_features=prefix)



                    # ref_text, ref_text_features, ref_img_features, hyp_text_features = _get_reference_text_image_features(
                    #     labels=tokens, image_features=prefix, preds=hyp_text)

                    # print(outputs.hidden_states[0].shape)  # torch.Size([50, 149, 1280])
                    # print(decoder_hidden_states.shape) # torch.Size([50, 63, 1280])
                    # print(projected_pred_feat.shape) # torch.Size([50, 640])
                    # print(outputs.projected_pred_feat.shape)

                    if torch.isnan(hyp_text_features).any():
                        print('val hyp_text_features is nan:', idx)
                    if torch.isnan(ref_text_features).any():
                        print('val ref_text_features is nan:', idx)
                    if torch.isnan(ref_img_features).any():
                        print('val ref_img_features is nan:', idx)

                    # ref_text, ref_text_features, ref_img_features = _get_reference_text_image_features(
                    #     labels=tokens, image_features=prefix)
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

        loss_per_epoch_val.append(val_loss / len(val_dataloader))
        loss_lm_per_epoch_val.append(val_loss_lm / len(val_dataloader))
        loss_contras_per_epoch_val.append(val_loss_contra / len(val_dataloader))
        print('val_loss_per_epoch: ', loss_per_epoch_val)
        print('val_loss_lm_per_epoch: ', loss_lm_per_epoch_val)
        print('val_loss_contras_per_epoch: ', loss_contras_per_epoch_val)

        if args.run_wandb:
            wandb.log({
                'train_loss_final': loss_per_epoch_train[-1],
                'train_loss_lowest': min(loss_per_epoch_train),
                # 'train_loss_lowest_index': loss_per_epoch_train.index(min(loss_per_epoch_train)),
                'valid_loss_final': loss_per_epoch_val[-1],
                'valid_loss_lowest': min(loss_per_epoch_val),
                # 'valid_loss_lowest_index': loss_per_epoch_val.index(min(loss_per_epoch_val)),
            })

        with open(os.path.join(output_dir, f"loss_per_epoch.json"), 'w') as f:
            json.dump({'train': loss_per_epoch_train, 'val': loss_per_epoch_val}, f)

    return model




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/sis_RN50x4_train.pkl')
    parser.add_argument('--val_data', default='/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/sis_RN50x4_val.pkl')
    parser.add_argument('--out_dir', default='/hpc/shared/uu_vl/yingjin_datasets/vist_data/clipcap_models/checkpoints/')
    parser.add_argument('--fn_prefix', default='vist_prev_sent', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=30)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='transformer', choices=('mlp', 'transformer'))
    parser.add_argument('--lang_model', type=str, default='gpt2', choices=('gpt2', 'gpt2-medium', 'gpt2-xl', 'opt'))
    parser.add_argument('--mask_type', type=str, default='ones', choices=('ones', 'default'))
    parser.add_argument('--num_layers', type=int, default=8)
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
    parser.add_argument('--start_contrastive_at_epoch', type=int, default=0)
    parser.add_argument('--run_gemini', dest='run_gemini', action='store_true')

    args = parser.parse_args()
    prefix_length = args.prefix_length  # length of input caption sequence

    train_dataset = ClipVistDataset(args.train_data, prefix_length, normalize_prefix=args.normalize_prefix, text_first=args.text_first)
    val_dataset = ClipVistDataset(args.val_data, prefix_length, normalize_prefix=args.normalize_prefix, text_first=args.text_first)

    if args.train_data.find('RN50') != -1:
        args.is_rn = True
    else:
        args.is_rn = False
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

    # for name, param in model.named_parameters():
    #     if name == 'projection_layer':
    #         print("projection_layer: ", param.requires_grad)
    #         sys.exit(0)
    # sys.exit(0)



    # print('Number of trainable named parameters is',
    #       sum(para.numel() for name, para in model.named_parameters() if para.requires_grad))
    #
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    #
    print('Number of trainable parameters is',
          sum(para.numel() for para in model.parameters() if para.requires_grad))

    train(train_dataset, val_dataset, model, args, output_dir=args.out_dir, output_prefix=args.fn_prefix)




if __name__ == '__main__':
    main()