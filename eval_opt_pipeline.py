# load modules
import os
from tkinter import ANCHOR
from eval import DocumentRetrieval
from tqdm import tqdm
import time
import json
import socket
import argparse
from datetime import datetime
from collections import defaultdict
from functools import partial
from multiprocessing.pool import ThreadPool
from multiprocessing.dummy import Pool
from threading import Thread

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

# pipelines
from dr.document_retrieval import DocRetrieval
from ss.train_ss import cleanse_and_split
from wiki_downloader import make_request

# pipelines for ss, rte model
from pipeline.ss_org import SentenceSelection
from pipeline.ss_knn import NewSSDataset, NewSentenceSelection
# from pipeline.ss_matrix import SentenceSelection, TensorStack
from pipeline.ss_matrix import SentenceSelection as Test
from pipeline.ss_matrix import TensorStack
from pipeline.rte import RecognizeTextualEntailment


# dr
# we'll not gonna use the wiki donwloader, rather use the pre-downloaded docs
class SimpleDR:
    def __init__(self, args):
        self.args = args
        with open(os.path.join(os.getcwd(), 'data/wiki/wiki_docs.json'), 'r') as f:
            self.wiki_docs = json.load(f)
        self.existing_titles = list(self.wiki_docs.keys())

    def doc_retrieval(self, title):

        documents = self.wiki_docs[title]
        date = datetime.strptime(self.args.claim_date, "%Y-%m-%d %H:%M:%S.%f")
        doc_dates = [datetime.strptime(dt, "%Y-%m-%dT%H:%M:%SZ") for dt in documents.keys()]
        doc_dates = [dt for dt in doc_dates if dt <= date]
        if not doc_dates:
            return
        text = documents[max(doc_dates).strftime("%Y-%m-%dT%H:%M:%SZ")] # 가장 최신 문서로 가져와라
        #print(text)
        self.dr_docs[title] = text

       # dr_titles => dr_docs ()

    def get_dr_results(self, claim):
        self.dr_docs = defaultdict(dict)
        get_titles = DocRetrieval(k_wiki_results=self.args.dr_k_wiki, tagger=self.args.dr_tagger,
                            parallel=self.args.parallel, n_cpu=self.args.n_cpu, production=True)

        _, dr_titles = get_titles.get_doc_for_claim(claim)
        dr_existing_titles = [title for title in dr_titles if title in self.existing_titles]
        
        pool = Pool(processes=6)
        pool.map(self.doc_retrieval, dr_existing_titles)
        pool.close()
        pool.join()

        dr_docs = self.dr_docs
        return dr_docs
    

class SimpleDR2:
    def __init__(self, claim_id, claim_date):
        #self.args = args
        self.claim_id = claim_id
        self.claim_date = claim_date
        with open(os.path.join(os.getcwd(), 'data/wiki/wiki_docs.json'), 'r') as f:
            self.wiki_docs = json.load(f)
        with open(os.path.join(os.getcwd(), 'dr/dr_results.json'), 'r') as f:
            self.dr_results = json.load(f)

    def doc_retrieval(self, title):
        try: 
            documents = self.wiki_docs[title]
        except:
            return
        date = datetime.strptime(self.claim_date, "%Y-%m-%d %H:%M:%S.%f")
        doc_dates = [datetime.strptime(dt, "%Y-%m-%dT%H:%M:%SZ") for dt in documents.keys()]
        doc_dates = [dt for dt in doc_dates if dt <= date]
        if not doc_dates:
            return
        text = documents[max(doc_dates).strftime("%Y-%m-%dT%H:%M:%SZ")] # 가장 최신 문서로 가져와라
        #print(text)
        self.dr_docs[title] = text

       # dr_titles => dr_docs ()

    def get_dr_results(self, claim):
        self.dr_docs = defaultdict(dict)
        self.titles = self.dr_results[self.claim_id] # a list     
        pool = Pool(processes=6)
        pool.map(self.doc_retrieval, self.titles)
        pool.close()
        pool.join()

        dr_docs = self.dr_docs
        return dr_docs
        
def main_worker(params):
    shared_dic = params[0]
    shared_dic['']
    
    claim_id 
    claim_id = claim_id
    claim_date = claim_file[claim_id]['Date']




def main(args):
    label_dict = {0: 'True', 1: 'False', 2: 'None'}

    # choose pipelines
    # 1. dr_pipeline
    if args.dr_pipeline == 0:
        dr_pipeline = DocumentRetrieval(args)
    elif args.dr_pipeline == 1:
        dr_pipeline = SimpleDR(args)
    elif args.dr_pipeline == 2:
        dr_pipeline = SimpleDR2(args)

    # 2. ss_pipeline 
    if args.ss_pipeline == 0:
        ss_pipeline = SentenceSelection(args)
    elif args.ss_pipeline == 1:
        ss_pipeline = NewSentenceSelection(args)
    elif args.ss_pipeline == 2:
        ss_pipeline = Test(args)

    # 3. rte_pipeline
    rte_pipeline = RecognizeTextualEntailment(args) # args.rte_pipeline 

    # make a claim dataset
    with open(os.path.join(args.cwd, 'data/wiki_claims.json'), 'r') as f:
        claim_file = json.load(f)
    with open(os.path.join(args.cwd, 'data/train_val_test_ids.json'), 'r') as f:
        ids = json.load(f)
    
    test_ids = ids['test_ids']
    claims = [claim_file[i]['claim'] for i in test_ids if i != '6032']
    labels = [claim_file[i]['True_False'] for i in test_ids if i != '6032']
    total = len(test_ids)
    corrects = 0
    
    # for further analysis
    pd_labels = []
    pbar = tqdm(total=len(test_ids), desc="Iteration")

    for k, i in enumerate(test_ids): # store the test_id and restart there dict: {'ids': ids, 'pd_label': pd_labels}
        if i == '6032':
            continue
        claim = claim_file[i]['claim']
        args.claim = claim
        args.label = claim_file[i]['True_False']
        if args.dr_pipeline == 1 or args.dr_pipeline == 2:
            args.claim_date = claim_file[i]['Date']
            args.claim_id = i        

    # ===== the claim is now selected ==== #
        print("Claim:", args.claim)

        start = time.time()
        # DR
        dr_results = dr_pipeline.get_dr_results(args.claim)
        dr_end = time.time()
        # if args.dr_pipeline == 1:
        #     dr_pipeline.save_titles_not_found()
        print("\n========== DR ==========")
        print(f"DR results: {', '.join(dr_results)}")
        print(f"DR Time taken: {dr_end - start:0.2f} (sec)")

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
        pd_labels.append(predicted_label)


        if args.label:
            if args.label == label_dict[predicted_label]:
                corrects += 1
                print("\nCorrect!!")
            else:
                print(f"\nIncorrect!! The correct label is {args.label}")
            print(f"temp accuracy: {round(corrects/(k+1), 5)}")
        pbar.update(1)
   
   # ================
    print("========== EVAL IS DONE ==========")
    print(f'test_accuracy: \n{round(corrects/total, 5)}')
    # recall rate / precision rate
    
    with open(os.path.join(os.getcwd(), f'pd_labels_{args.ss_pipeline}.pkl'), 'wb') as f:
        pickle.dump(pd_labels, f)

# argument

if __name__ == "__main__":
    print(f"Job is running on {socket.gethostname()}")

    parent_parser = argparse.ArgumentParser(add_help = False)
    parent_parser.add_argument("--claim",
                        default="한국 최초의 메이저리그 선수인 박찬호는 은퇴 후 돈을 벌기 위해 미국 내셔널리그 샌디에이고 파드리스의 고문이다.",
                        type=str,
                        help="claim sentence to verify")
    parent_parser.add_argument("--label",
                        default="True",
                        type=str,
                        help="gold label, 'True', 'False', or 'NEI'")
    parent_parser.add_argument("--dr_k_wiki",
                        type=int,
                        default=3,
                        help="first k pages for wiki search")
    parent_parser.add_argument("--dr_tagger",
                        type=str,
                        default="Okt",
                        help="KoNLPy tagger. Strongly recommend to use the default.")
    parent_parser.add_argument("--non_parallel",
                        default=False,
                        action="store_true",
                        help="Do not use multiprocessing for downloading documents through mediawiki API")
    parent_parser.add_argument("--n_cpu",
                        default=None,
                        type=int,
                        help="Number of cpus to utilize for multiprocessing")
    parent_parser.add_argument("--cwd",
                        default = os.getcwd(),
                        type = str,
                        help = "current working directory")
    parent_parser.add_argument("--top_k",
                        default = 5,
                        type = int,
                        help = "The number of selected sentences")

    # baseline options
    parent_parser.add_argument("--dr_pipeline",
                        default = 0,
                        type = int,
                        help = "dr pipeline type") #0: original #1: simple_dr #2: simple_dr2
    parent_parser.add_argument("--ss_pipeline",
                        default = 0,
                        type = int,
                        help = "ss pipeline type") #0: original #1: new_ss #2: matrix
    parent_parser.add_argument("--ss_batchsize",
                            default=8,
                            type=int,
                            help="Batch size for validation examples.")
    parent_parser.add_argument("--rte_pipeline", #0: original(cls, spe) #1: best(cls, mpe, ev_num 4) #2: (rgs, mpe, ev_num 4, nei_scale 0.45), #3: (rgs, spe, nei_scale 0.45)
                                default = 1,
                                type = int,
                                help = "rte pipline type")
    parent_parser.add_argument("--rte_cache_dir",
                            default="./data/models/",
                            type=str,
                            help="Where the pre-trained models for RTE will be / is stored")
    parent_parser.add_argument("--rte_model",
                            default="koelectra",
                            type=str,
                            help='"koelectra" if want to use KoElectra model (https://github.com/monologg/KoELECTRA).')
    parent_args = parent_parser.parse_args()

    # detail options accordingly
    parser = argparse.ArgumentParser(parents = [parent_parser])
    if parent_args.ss_pipeline == 0:                            
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

    elif parent_args.ss_pipeline == 1:
        parser.add_argument("--new_ss_model_dir",
                            default = "./new_ss/",
                            type = str,
                            help = "Where new SS model is stored")
        parser.add_argument("--new_ss_model",
                            default = "tuned_model.pt",
                            type = str,
                            help = "New SS model name")
    elif parent_args.ss_pipeline == 2:
        parser.add_argument("--ss_cache_dir",
                            default="./data/models/",
                            type=str,
                            help="Where the pre-trained models for SS will be / is stored")
    #     #parser.add_argument("")

    rte_checkpoints_dirs = ['./rte/checkpoints/', './new_rte/checkpoints/']
    rte_checkpoints = ['', '_cls_mpe_4_noised', '_rgs_mpe_4', '_rgs_spe']
    rte_checkpoint_dir = rte_checkpoints_dirs[0] if parent_args.rte_pipeline == 0 else rte_checkpoints_dirs[1]
    rte_checkpoint = rte_checkpoints[parent_args.rte_pipeline]

    parser.add_argument('--rte_checkpoints_dir',
                        default = f'{rte_checkpoint_dir}',
                        type = str,
                        help='Where checkpoints for RTE will be / is stored.')
    parser.add_argument('--rte_checkpoint',
                        default = f'best_ckpt{rte_checkpoint}.pth',
                        help='RTE checkpoint file name.')

    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.parallel = ~(args.non_parallel)
    args.n_cpu = args.n_cpu if args.parallel else 1
    #args.max_length = 1024 if args.ss_pipeline == 1 and args.top_k > 7 else 512
    #if args.rte_model == "koelectra":
    #    args.rte_checkpoints_dir = os.path.dirname(args.rte_checkpoints_dir) + "_" + args.rte_model

    print(args.rte_checkpoint)
    main(args)