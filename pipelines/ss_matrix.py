# import modules
# load modules
import os
from tkinter import ANCHOR
from tqdm import tqdm
import time
import json
import socket
import argparse
import numpy as np
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
# for multi-gpu training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import warnings
from ss.train_ss import cleanse_and_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SentenceSelection:
    #@profile
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',
                                                          cache_dir=args.ss_cache_dir, num_labels=2)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        #ckpt_path = os.path.join(args.ss_checkpoints_dir, args.ss_checkpoint)
        #checkpoint = torch.load(ckpt_path, map_location = torch.device('cpu'))
        #self.model.load_state_dict(checkpoint["state_dict"])

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

        ### === modified === ###
        sentences = candidates + [claim]
        self.size = len(sentences)
        self.matrix_indices = []

        for i in tqdm(range(1, self.size)):
            sub_input_ids = []
            sub_segment_ids = []
            sub_input_masks = []
            for k in range(i):
                if len((set(sentences[i].split()) & set(sentences[k].split()))) != 0: # if there exists common unigrams more than one
                    sentence_a = self.tokenizer.tokenize(sentences[i])
                    sentence_b = self.tokenizer.tokenize(sentences[k])
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

                    sub_input_ids.append(input_ids)
                    sub_segment_ids.append(segment_ids)
                    sub_input_masks.append(mask)
                    self.matrix_indices.append(set(i,k))

            if len(sub_input_ids) != 0:
                all_input_ids.extend(sub_input_ids)
                all_segment_ids.extend(sub_segment_ids)
                all_input_masks.extend(sub_input_masks)

        # torch TensorDataset
        all_input_ids = torch.LongTensor(all_input_ids)
        all_segment_ids = torch.LongTensor(all_segment_ids)
        all_input_masks = torch.LongTensor(all_input_masks)
        ss_dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_masks)
        return titles, candidates, ss_dataset

        #@profile
    def validate(self, gpu, ss_dataset):
        #self.args.gpu = gpu
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:9856", rank=gpu, world_size=self.args.n_gpu)
        model = self.model
        model = model.to(gpu)
        model = DDP(model, device_ids = [gpu])
        dist.barrier()
        
        model.eval()
        ss_dataloader = DataLoader(ss_dataset, batch_size=self.args.ss_batchsize, shuffle=False)

        if gpu == 0:
            print("=== Running validation ===")
            pbar = tqdm(total = len(ss_dataloader), desc = "Iteration")

        logits = []

        for batch in ss_dataloader:
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, segment_ids, input_mask = batch
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
            
            logits.append(outputs.logits)
        
            if gpu == 0:
                pbar.update(1)
        
        logits_cat = torch.cat(logits, dim=0)
        self.calculate_metric(logits_cat)
        dist.barrier()

    def calculate_metric(self, logits_cat):        
        cnt = 0
        logits_matrix = [[0]*i for i in range(0,self.size)]
        for n, i in enumerate(self.matrix_indices):
            tmp = list(i)
            tmp.sort()
            logits_matrix[tmp[0]][tmp[1]] = logits_cat[n]

        #logits_matrix = [logits_cat[(i-1)*(i)/2:(i)*(i+1)/2] for i in range(1, self.size-1)]

        ### === moidfied === ###
        self.top5_indices = []
        self.scores = []
        group_id = self.size-1
        tensor_stack = TensorStack(self.args)
        tensor_stack.push(logits_matrix[-1])

        # find top1 index and logit val in Group-n
        print("matrix loaded")
        
        for i in range(self.args.top_k):
            score, idx = tensor_stack.stack.topk(1)

            #logit_idx = (group_id)*(group_id+1)/2 + idx
            tensor_stack.pop(score)
            new_cands = logits_matrix[idx] + [logits_matrix[k][idx] for k in range(idx, self.size)]
            assert len(new_cands) == self.size - 1
            tensor_stack.push(new_cands*score)

            self.scores.append(score)
            self.top5_indices.append(idx)

            print(f"found {i}th cand")
        self.scores = F.softmax(self.scores, 1)[:,1]

        #return scores, top_indices

        # scores, top5_indices = softmax_logits.topk(min(5, len(softmax_logits)))
        # return scores, top5_indices
    #@profile

    def get_results(self, claim, dr_results):
        titles, candidates, ss_dataset = self.build_dataset(claim, dr_results, max_length=512)
        self.args.n_gpu = torch.cuda.device_count()
        mp.spawn(self.validate, nprocs = self.args.n_gpu,
                args = (ss_dataset,))
        #scores, top5_indices = self.validate(ss_dataset)
        return (
            self.scores,
            [titles[idx] for idx in self.top5_indices],
            [candidates[idx] for idx in self.top5_indices]
        )

class TensorStack:
    def __init__(self, args):
        self.args = args
        self.stack = []
    
    def push(self, tensors):
        self.stack.append(tensors)
        self.stack = torch.cat(self.stack, dim = 0)

    def pop(self, indices):
        mask = torch.one(self.stack.numel(), dtype = torch.bool)
        mask[indices] = False
        self.stack = self.stack[mask]