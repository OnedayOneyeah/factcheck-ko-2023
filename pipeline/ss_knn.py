# import modules
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

class NewSSDataset(Dataset):
    def __init__(self, dataset, sent_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length = max_len, pad = pad, pair = pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
    def __getitem__(self, i):
        return (self.sentences[i])
    def __len__(self):
        return (len(self.sentences))


class NewSentenceSelection:
    def __init__(self, args):
        self.args = args
        _, vocab = get_pytorch_kobert_model()
        tokenizer_bert = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer_bert, vocab, lower = False)
        self.tokenizer = tok
        self.model = torch.load(os.path.join(args.new_ss_model_dir, args.new_ss_model))
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def get_embeddings(self, claim, candidates):
        #claim_embeddings = []
        #ss_embeddings = []
        transform = nlp.data.BERTSentenceTransform(self.tokenizer, max_seq_length = 64, pad = True, pair = True)
        max_len = 64

        # dataset
        claim_dataset = NewSSDataset(claim, 0, self.tokenizer, max_len, pad =True, pair = False)
        cands_dataset = NewSSDataset(candidates, 0, self.tokenizer, max_len, True, False)
        
        # dataloader
        claim_dataloader = torch.utils.data.DataLoader(claim_dataset, batch_size = 1, num_workers = 5)
        cands_dataloader = torch.utils.data.DataLoader(cands_dataset, batch_size = 10, num_workers = 5)
        
        # get embeddings
        self.model.eval()

        # for claim
        self.model.extracter.clear()
        
        for batch_id, (token_ids, valid_length, segment_ids) in enumerate(claim_dataloader):
            token_ids = token_ids.long().to(self.args.device)
            segment_ids = segment_ids.long().to(self.args.device)
            valid_length = valid_length
            with torch.no_grad():
                sequence = self.model(token_ids, valid_length, segment_ids)
        claim_embeddings = self.model.extracter.embeddings[0][0].reshape(768,1)
        #print(claim_embeddings)
        
        # candidates
        self.model.extracter.clear()
        for batch_id, (token_ids, valid_length, segment_ids) in enumerate(cands_dataloader):
            token_ids = token_ids.long().to(self.args.device)
            segment_ids = segment_ids.long().to(self.args.device)
            valid_length = valid_length
            with torch.no_grad():
                sequence = self.model(token_ids, valid_length, segment_ids)
        #print(f'=====length of embedding list: {len(self.model.extracter.embeddings)}')
        #print(f'=====embedding size: {self.model.extracter.embeddings[0].size()}')
        #ss_embeddings
        for i in range(len(self.model.extracter.embeddings)):
            if i == 0:
                ss_embeddings = self.model.extracter.embeddings[0]
            else:
                ss_embeddings = torch.cat((ss_embeddings, self.model.extracter.embeddings[i]), 0)
        print(f'=====ss embedding size: {ss_embeddings.size()}')
        #print(f'=====claim size: {claim_embeddings.size()}')
        self.model.extracter.clear()
        #print(f'claim embeddings: {claim_embeddings}\n')
        #print(f'cand embeddings: {ss_embeddings}\n')
        
        return claim_embeddings, ss_embeddings

        
    def build_dataset(self, claim, dr_docs, max_length = 256): # 512 => 256, since we don't concat two ss
        titles = []
        candidates = []

        for title in dr_docs:
            text = dr_docs[title]
            text_lst = cleanse_and_split(text)
            candidates.extend(text_lst)
            titles.extend([title]*len(text_lst))

        self.claim = [claim]
        self.candidates = [cand for cand in candidates]
        assert len(titles) == len(candidates)

        return titles, candidates

    def validate(self, claim_embeddings, ss_embeddings):
        top_k = self.args.top_k
        
        #print(f"Claim embeddings:\n{claim_embeddings}, ss_embeddings:\n{ss_embeddings_chunk}")
        
        vals = torch.matmul(ss_embeddings, claim_embeddings)
        #print(f'claim len: {len(claim_embeddings)}\nss len: {len(ss_embeddings)}\nvals len: {len(vals)}')
        #print(f'vals: \n{vals}')
        
        # pick top 5 scores
        #print(f'*vals: {vals}')
        softmax_embeddings = F.softmax(vals, dim = 0)[:, 0]
        #print(f'*softmax_embeddings: {softmax_embeddings}')
        scores, topk_indices = softmax_embeddings.topk(min(top_k, len(softmax_embeddings)))

        return scores, topk_indices

    def get_results(self, claim, dr_results):

        # build dataset
        titles, candidates = self.build_dataset(claim, dr_results)

        # get embeddings
        claim_embeddings, ss_embeddings = self.get_embeddings(self.claim, self.candidates)

        # validate
        scores, topk_indices = self.validate(claim_embeddings, ss_embeddings)

        return(
            scores,
            [titles[idx] for idx in topk_indices],
            [candidates[idx]for idx in topk_indices]
        )