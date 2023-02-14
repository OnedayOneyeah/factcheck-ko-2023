# import modules

import os
import time
import json
import socket
import argparse
from tqdm.notebook import tqdm
from datetime import datetime
from collections import defaultdict
from functools import partial
from multiprocessing.dummy import Pool

import torch
import torch.nn.functional as F
from torch.utils.data import (
    TensorDataset,
    DataLoader,
)
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    ElectraTokenizerFast,
    ElectraForSequenceClassification
)

from dr.document_retrieval import DocRetrieval
from ss.train_ss import cleanse_and_split
from wiki_downloader import make_request

# load test data for claims
with open('./data/wiki_claims.json', 'r') as f:
    claim_data = json.load(f)
with open('./data/train_val_test_ids.json', 'r') as f:
    ids = json.load(f)
    test_ids = ids['test_ids'] #list


# make a test_tbl dict to prepare the dataset

claims = []
labels = []
test_tbl = dict()
for k, i in enumerate(test_ids):
    if i == '6032':
        pass
    else:
        test_tbl[i] = {'claim': claim_data[test_ids[k]]['claim'],
        'True_False' : claim_data[test_ids[k]]['True_False'], 'is_correct': -1, 'rte_score': -1.0}

# eval.py

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
        for batch in ss_dataloader:
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
        self.model.to("cpu")

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

    # /// new lines ///
    claims_to_test = list(test_tbl.keys())

    dr_pipeline = DocumentRetrieval(args)
    ss_pipeline = SentenceSelection(args)
    rte_pipeline = RecognizeTextualEntailment(args)

    for i in tqdm(claims_to_test):

        args.claim = test_tbl[i]['claim']

        #dr_pipeline. #args_claim 바꿔주기
        #ss_pipeline
        rte_pipeline

        
        print("Claim:", args.claim)
    

        start = time.time()
        # DR
        dr_results = dr_pipeline.get_dr_results(args.claim)
        dr_end = time.time()
        print("\n========== DR ==========")
        print(f"DR results: {', '.join(dr_results)}")
        print(f"DR Time taken: {dr_end - start:0.2f} (sec)")

        # SS
        ss_scores, ss_titles, ss_results = ss_pipeline.get_results(args.claim, dr_results)
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
        test_tbl[i]['rte_score'] = rte_score
        rte_end = time.time()
        print("\n========== RTE ==========")
        print(f"Predicted Label: {label_dict[predicted_label]} ({rte_score})")
        print(f"RTE Time taken: {rte_end - ss_end:0.2f} (sec)")

        if args.label:
            if args.label == label_dict[predicted_label]:
                test_tbl[i]['is_correct'] = 1
                print("\nCorrect!!")
            else:
                test_tbl[i]['is_correct'] = 0
                print("\nIncorrect!!")


if __name__ == "__main__":
    print(f"Job is running on {socket.gethostname()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--claim",
                        default="앤드류 응은 홍콩에서 태어났다.",
                        type=str,
                        help="claim sentence to verify")
    parser.add_argument("--label",
                        default="False",
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
