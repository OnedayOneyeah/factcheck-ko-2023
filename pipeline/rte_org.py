#import modules
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

# =====from here, for new ss model=====
import torch.optim as optim
import gluonnlp as nlp
import numpy as np

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from sklearn.model_selection import train_test_split
import random
from new_ss.tuned_kobert import GetEmbeddings, BERTDataset, BERTClassifier, BuildTrainModel

from transformers.optimization import get_cosine_schedule_with_warmup
import pickle
import warnings
warnings.filterwarnings("ignore")
# ======================================


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


class RecognizeTextualEntailment:
    #@profile
    def __init__(self, args):
        self.args = args
        if self.args.rte_model == "koelectra":
            self.tokenizer = ElectraTokenizerFast.from_pretrained("monologg/koelectra-base-v3-discriminator")
            self.model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator",
                                                                          cache_dir=args.rte_cache_dir, num_labels=3)
        else:
            self.tokenizer = BertTokenizerFast
            self.model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',
                                                                       cache_dir=args.rte_cache_dir, num_labels=3)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = os.path.join(args.rte_checkpoints_dir, args.rte_checkpoint)
        checkpoint = torch.load(ckpt_path, map_location = torch.device('cpu'))
        self.model.load_state_dict(checkpoint["state_dict"])

    #@profile
    def build_dataset(self, claim, ss_results, max_length=512):
        all_input_ids = []
        all_segment_ids = []
        all_input_masks = []
        for evidence in ss_results:
            sentence_a = self.tokenizer.tokenize(evidence)
            sentence_b = self.tokenizer.tokenize(claim)
            if len(sentence_a) + len(sentence_b) > max_length - 3:
                diff = (len(sentence_a) + len(sentence_b)) - (max_length - 3)
                sentence_a = sentence_a[:-diff]

            tokens = ["[CLS]"] + sentence_a + ["[SEP]"] + sentence_b + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * (len(sentence_a) + 2) + [1] * (len(sentence_b) + 1)
            mask = [1] * len(input_ids)

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

        all_input_ids = torch.LongTensor(all_input_ids)
        all_segment_ids = torch.LongTensor(all_segment_ids)
        all_input_masks = torch.LongTensor(all_input_masks)
        dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_masks)
        return dataset

    #@profile
    def validate(self, rte_dataset, mode="sum"):
        self.model.eval()
        rte_dataloader = DataLoader(rte_dataset, batch_size=len(rte_dataset), shuffle=False)

        logits = []
        for batch in rte_dataloader:
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
        softmax_logits = F.softmax(logits_cat, dim=1)
        pred_for_each_claim = softmax_logits.argmax(dim=1)

        score = None
        if mode == "fever":
            if 0 in pred_for_each_claim:
                pred = 0
            else:
                if 1 in pred_for_each_claim:
                    pred = 1
                else:
                    pred = 2
        elif mode == "majority_vote":
            pred = int(pred_for_each_claim.mode().values)
        elif mode == "sum":
            pred = softmax_logits.sum(dim=0).argmax().item()

            # score is average of softmax_logits
            score = softmax_logits.mean(dim=0).max().item()
        return pred, score

    #@profile
    def get_results(self, claim, ss_results):
        rte_dataset = self.build_dataset(claim, ss_results, max_length=512)
        pred, score = self.validate(rte_dataset)
        return pred, score
