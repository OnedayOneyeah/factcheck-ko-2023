# import modules
# load modules
import os
from tkinter import ANCHOR
from tqdm import tqdm
import time
import json
import socket
import argparse
from datetime import datetime
from collections import defaultdict
from functools import partial
from multiprocessing.dummy import Pool
import torch
import torch.nn.functional as F
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    Dataset
)
from transformers import (
    AdamW,
    BertTokenizerFast,
    BertForSequenceClassification,
    ElectraTokenizerFast,
    ElectraForSequenceClassification
)

from ss.train_ss import cleanse_and_split


class SentenceSelection:
    #@profile
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',
                                                          cache_dir=args.ss_cache_dir, num_labels=2)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = os.path.join(args.ss_checkpoints_dir, args.ss_checkpoint)
        checkpoint = torch.load(ckpt_path, map_location = torch.device('cpu'))
        self.model.load_state_dict(checkpoint["state_dict"])

    #@profile
    def build_dataset(self, claim, dr_docs, max_length=512):
        titles = []
        candidates = []
        for title in dr_docs:
            text = dr_docs[title]
            text_lst = cleanse_and_split(text)
            candidates.extend(text_lst)
            titles.extend([title]*len(text_lst))
        candidates = [cand for cand in candidates]

        assert len(titles) == len(candidates)

        all_input_ids = []
        all_segment_ids = []
        all_input_masks = []
        for cand in candidates:
            sentence_a = self.tokenizer.tokenize(cand)
            sentence_b = self.tokenizer.tokenize(claim)
            if len(sentence_a) + len(sentence_b) > max_length - 3:
                diff = (len(sentence_a) + len(sentence_b)) - (max_length - 3)
                sentence_a = sentence_a[:-diff]

            tokens = ["[CLS]"] + sentence_a + ["[SEP]"] + sentence_b + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * (len(sentence_a) + 2) + [1] * (len(sentence_b) + 1)
            mask = [1] * len(input_ids)

            # Zero-padding
            padding = [0] * (max_length - len(input_ids))
            input_ids += padding
            segment_ids += padding
            mask += padding
            assert len(input_ids) == max_length
            assert len(segment_ids) == max_length
            assert len(mask) == max_length

            all_input_ids.append(input_ids)
            all_segment_ids.append(segment_ids)
            all_input_masks.append(mask)

        # torch TensorDataset
        all_input_ids = torch.LongTensor(all_input_ids)
        all_segment_ids = torch.LongTensor(all_segment_ids)
        all_input_masks = torch.LongTensor(all_input_masks)
        ss_dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_masks)
        return titles, candidates, ss_dataset

    #@profile
    def validate(self, ss_dataset):
        self.model.eval()
        ss_dataloader = DataLoader(ss_dataset, batch_size=self.args.ss_batchsize, shuffle=False)

        logits = []
        for batch in tqdm(ss_dataloader):
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, segment_ids, input_mask = batch
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
            logits.append(outputs.logits)

        logits_cat = torch.cat(logits, dim=0)
        #print(f'logits_cat: {logits_cat}')
        softmax_logits = F.softmax(logits_cat, 1)[:, 1]
        #print(f'softmax_logits: {softmax_logits}')
        # pick top 5 sentences
        scores, top5_indices = softmax_logits.topk(min(5, len(softmax_logits)))
        return scores, top5_indices

    #@profile
    def get_results(self, claim, dr_results):
        titles, candidates, ss_dataset = self.build_dataset(claim, dr_results, max_length=512)
        scores, top5_indices = self.validate(ss_dataset)
        return (
            scores,
            [titles[idx] for idx in top5_indices],
            [candidates[idx] for idx in top5_indices]
        )