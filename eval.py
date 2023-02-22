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

from dr.document_retrieval import DocRetrieval
from ss.train_ss import cleanse_and_split
from wiki_downloader import make_request


class DocumentRetrieval:
    #@profile
    def __init__(self, args):
        self.args = args
        self.dr_model = DocRetrieval(k_wiki_results=args.dr_k_wiki, tagger=args.dr_tagger,
                            parallel=args.parallel, n_cpu=args.n_cpu, production=True)

    #@profile
    def wiki_download(self, dr_titles):
        dr_docs = defaultdict(dict)
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        multiproc_partial_make_request = partial(make_request, date=date, titles_annotated=[], corpus=dr_docs)
        with Pool(processes=self.args.n_cpu) as p:
            for doc_text, _ in (p.imap if self.args.parallel else map)(
                    multiproc_partial_make_request, dr_titles
            ):
                if doc_text:
                    title, _, text = doc_text
                    dr_docs[title] = text
        return dr_docs

    #@profile
    def get_dr_results(self, claim):
        _, dr_titles = self.dr_model.get_doc_for_claim(claim)
        dr_docs = self.wiki_download(dr_titles)
        #print(dr_docs)
        return dr_docs


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
        softmax_logits = F.softmax(logits_cat, 1)[:, 1]
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

ANCHOR
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
        claim_embeddings = self.model.extracter.embeddings[0][0].reshape(768,1).t()
        #print(claim_embeddings)
        
        # candidates
        self.model.extracter.clear()
        for batch_id, (token_ids, valid_length, segment_ids) in enumerate(cands_dataloader):
            token_ids = token_ids.long().to(self.args.device)
            segment_ids = segment_ids.long().to(self.args.device)
            valid_length = valid_length
            with torch.no_grad():
                sequence = self.model(token_ids, valid_length, segment_ids)
        #ss_embeddings
        for i in range(len(self.model.extracter.embeddings)):
            if i == 0:
                ss_embeddings = self.model.extracter.embeddings[0]
            else:
                ss_embeddings = torch.cat((ss_embeddings, self.model.extracter.embeddings[i]), 0)
        ss_embeddings = ss_embeddings.t()

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
        
        vals = torch.matmul(claim_embeddings, ss_embeddings)
        #print(f'claim len: {len(claim_embeddings)}\nss len: {len(ss_embeddings)}\nvals len: {len(vals)}')
        #print(f'vals: \n{vals}')
        
        # pick top 5 scores
        softmax_embeddings = F.softmax(vals, dim = 1)
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


def main(args):
    label_dict = {0: 'True', 1: 'False', 2: 'NEI'}

    dr_pipeline = DocumentRetrieval(args)
    if args.ss_pipeline == 0:
        ss_pipeline = SentenceSelection(args)
    elif args.ss_pipeline == 1:
        ss_pipeline = NewSentenceSelection(args)
    rte_pipeline = RecognizeTextualEntailment(args)

    # ===== we'll select claims!!! ====== #
    with open('/home/yewon/factcheck_automization/data/wiki_claims.json', 'r') as f:
        claim_file = json.load(f)
    with open('/home/yewon/factcheck_automization/data/train_val_test_ids.json', 'r') as f:
        ids = json.load(f)
    
    test_ids = ids['test_ids']
    total = len(test_ids)
    corrects = 0
    
    for i in tqdm(test_ids):
        claim = claim_file[i]['claim']
        args.claim = claim
    # ===== the claim is now selected ==== #

        print("Claim:", args.claim)

        start = time.time()
        # DR
        dr_results = dr_pipeline.get_dr_results(args.claim)
        dr_end = time.time()
        print("\n========== DR ==========")
        print(f"DR results: {', '.join(dr_results)}")
        print(f"DR Time taken: {dr_end - start:0.2f} (sec)")
        print(dr_results)

        # SS
        ss_scores, ss_titles, ss_results = ss_pipeline.get_results(claim = args.claim, dr_results = dr_results)
        ss_results_print = "\n".join([
            f"{ss_title}: {ss_result} ({ss_score})"
            for ss_title, ss_result, ss_score
            in zip(ss_titles, ss_results, ss_scores)])
        ss_end = time.time()
        print("\n========== SS ==========")
        print(f'SS results: \n{ss_results_print}')
        print(f"SS Time taken: {ss_end - dr_end:0.2f} (sec)")

        # RTE
        predicted_label, rte_score = rte_pipeline.get_results(args.claim, ss_results)
        rte_end = time.time()
        print("\n========== RTE ==========")
        print(f"Predicted Label: {label_dict[predicted_label]} ({rte_score})")
        print(f"RTE Time taken: {rte_end - ss_end:0.2f} (sec)")

        if args.label:
            if args.label == label_dict[predicted_label]:
                corrects += 1
                print("\nCorrect!!")
            else:
                print("\nIncorrect!!")
    
    print("========== EVAL IS DONE ==========")
    print(f'test_accuracy: \n{round(corrects/total, 5)}')


if __name__ == "__main__":
    print(f"Job is running on {socket.gethostname()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--claim",
                        default="에리히 폰 만슈타인은 제1차 세계 대전 당시 서부전선과 동부전선에서 복무했다.",
                        type=str,
                        help="claim sentence to verify")
    parser.add_argument("--label",
                        default="True",
                        type=str,
                        help="gold label, 'True', 'False', or 'NEI'")
    parser.add_argument("--dr_k_wiki",
                        type=int,
                        default=3,
                        help="first k pages for wiki search")
    parser.add_argument("--dr_tagger",
                        type=str,
                        default="Okt",
                        help="KoNLPy tagger. Strongly recommend to use the default.")
    
    parser.add_argument("--ss_cache_dir",
                        default="./data/models/",
                        type=str,
                        help="Where the pre-trained models for SS will be / is stored")
    parser.add_argument("--ss_checkpoints_dir",
                        default="./ss/checkpoints/", #./ss/12_04/checkpoints/
                        type=str,
                        help="Where checkpoints for SS will be / is stored.")
    parser.add_argument("--ss_checkpoint",
                        default="best_ckpt.pth", #temp_ckpt.pth
                        type=str,
                        help="SS checkpoint file name.")
    parser.add_argument("--ss_batchsize",
                        default=8,
                        type=int,
                        help="Batch size for validation examples.")
    # arguments for new ss model
    parser.add_argument("--new_ss_model_dir",
                        default = "./new_ss/",
                        type = str,
                        help = "Where new SS model is stored")
    parser.add_argument("--new_ss_model",
                        default = "tuned_model.pt",
                        type = str,
                        help = "New SS model name")
    parser.add_argument("--new_ss_batchsize",
                        default = 8,
                        type = int,
                        help = "Batch size for validation examples")
    parser.add_argument("--top_k",
                        default = 5,
                        type = int,
                        help = "The number of selected sentences")
    parser.add_argument("--ss_pipeline",
                        default = 0,
                        type = int,
                        help = "ss pipeline type") #0: original #1: new_ss
    parser.add_argument("--rte_cache_dir",
                        default="./data/models/",
                        type=str,
                        help="Where the pre-trained models for RTE will be / is stored")
    parser.add_argument("--rte_checkpoints_dir", 
                        default="./rte/checkpoints/", #./rte/12_04/checkpoints/
                        type=str,
                        help="Where checkpoints for RTE will be / is stored.")
    parser.add_argument("--rte_checkpoint",
                        default="best_ckpt.pth",
                        type=str,
                        help="RTE checkpoint file name.")
    parser.add_argument("--rte_model",
                        default="koelectra",
                        type=str,
                        help='"koelectra" if want to use KoElectra model (https://github.com/monologg/KoELECTRA).')
    parser.add_argument("--non_parallel",
                        default=False,
                        action="store_true",
                        help="Do not use multiprocessing for downloading documents through mediawiki API")
    parser.add_argument("--n_cpu",
                        default=None,
                        type=int,
                        help="Number of cpus to utilize for multiprocessing")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.parallel = ~(args.non_parallel)
    args.n_cpu = args.n_cpu if args.parallel else 1
    #if args.rte_model == "koelectra":
    #    args.rte_checkpoints_dir = os.path.dirname(args.rte_checkpoints_dir) + "_" + args.rte_model

    main(args)
