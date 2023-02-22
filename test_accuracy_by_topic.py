# ===== notes =====
# - 하려는 것
#     - test claim의 corresponding embedding, topic label, is_correct 가지고 토픽별 test accuracy 측정
# - 필요한 데이터
#     - test claim embedding
#     - test claim topic label
#     - test claim True_False
#     - test claim is_correct
# - 모델: tuned_kobert에 labels 추가해서 쓸거임

# import modules
from email.policy import default
import os
import json
import argparse
from collections import defaultdict
import random
import pickle
from tqdm import tqdm, tqdm_notebook
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
from sklearn.model_selection import train_test_split

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from new_ss.tuned_kobert import BERTDataset, GetEmbeddings, BERTClassifier, BuildTrainModel

def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx]))/a).item() * 100

def testModel(args, model, seq):
    cate = ["정치","경제","사회", "생활/문화","세계","기술/IT", "연예", "스포츠"]
    tmp = [seq]
    transform = nlp.data.BERTSentenceTransform(args.tok, args.max_len, pad=True, pair=False)
    tokenized = transform(tmp)
    print(f'tokenized: {tokenized}')

    model.eval()
    with torch.no_grad():
        testModel_result = model(torch.LongTensor([tokenized[0]]).to(args.device), [tokenized[1]], torch.LongTensor(tokenized[2]).to(args.device))
        idx = testModel_result.argmax().cpu().item()
    print("클레임 카테고리는:", cate[idx])
    print("신뢰도는:", "{:.2f}%".format(softmax(testModel_result,idx)))
    return (idx, model.extracter.embeddings)
    
    

def main(args):
    # load data
    with open('/home/yewon/factcheck_automization/data/wiki_claims.json', 'r') as f:
        wiki_claims = json.load(f)
        dataset = defaultdict(dict)
        
    with open('/home/yewon/factcheck_automization/data/train_val_test_ids.json', 'r') as f:
        train_val_test_ids = json.load(f)

    test_ids = train_val_test_ids['test_ids']

    if args.ss_pipeline == 0:
        with open(os.path.join(args.pd_labels_path, 'pd_labels_0.pkl'), 'rb') as f:
            pd_labels = json.load(f)
    elif args.ss_pipeline == 1:
        with open(os.path.join(args.pd_labels_path, 'pd_labels_15.pkl'), 'rb') as f:
            pd_labels = json.load(f)
    elif args.ss_pipeline == 2:
        with open(os.path.join(args.pd_labels_path,'pd_labels_110.pkl'), 'rb') as f:
            pd_labels = json.load(f)


    for i, id in enumerate(test_ids):
        claim = wiki_claims[id]['claim']
        true_false = wiki_claims[id]['True_False']
        idx, embedding = testModel(args, args.model, claim)
        dataset[id]['claim'] = claim
        dataset[id]['True_False'] = true_false
        dataset[id]['top'] = idx # topic label: 0~7
        dataset[id]['emb'] = embedding[0][0]
        dataset[id]['pre'] = pd_labels[i] # prediction from ss&rte models

        args.model.extracter.clear()

        

    #seq = '인어에 대한 전설은 전 세계 문화권에 두루 존재하고 있지만, 인어를 모르는 나라도 있다.'


    # claims = defaultdict(dict)
    # for cid in test_ids:
    #     claims[cid] = {'emb' :,
    #                     'topic_label' :,
    #                     'True_False' :,
    #                     'pd_label':}
        

    
    # load model

    

# get embeddings: by using pool
# visualization
# test accuracy by topic

#def visualize(args):


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # options
    parser.add_argument('--ss_pipeline',
                        default = '0',
                        type = int,
                        help = "choose the ss model of which you want to test the accuracy") #0: org, 1: new_ss(k=5), 2: new_ss(k=10), 3: new_ss(k=20)
    parser.add_argument('--visualize',
                        default = False,
                        action = "store_true",
                        help = "visualize the test accuracy by topic")
    args = parser.parse_args()
    args.pd_labels_path = '/home/yewon/factcheck_automization'
    args.model = torch.load('/home/yewon/factcheck_automization/new_ss/tuned_model.pt')
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer_bert = get_tokenizer()
    args.tok = nlp.data.BERTSPTokenizer(tokenizer_bert, vocab, lower = False)
    args.max_len = 64


# max_len = 64
# batch_size = 32
# warmup_ratio = 0.1
# num_epochs = 10
# max_grad_norm = 1
# log_interval = 200
# learning_rate = 5e-5

    main(args)

