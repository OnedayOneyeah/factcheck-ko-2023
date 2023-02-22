from tkinter import ANCHOR
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import argparse

import os

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from sklearn.model_selection import train_test_split
import random
import pickle
import warnings 
warnings.filterwarnings("ignore")


bertmodel, vocab = get_pytorch_kobert_model() #pretrained bert model 불러오기


# a class to load txt files
from tqdm import tqdm
import time

def load_files(args, shuffle):
    labels = ['정치', '경제', '사회', '생활/문화', '세계', '기술/IT', '연예', '스포츠'] 
    data = []
    target = []
    for i in tqdm(range(len(labels)), desc = 'outer', position = 0):
      for k in tqdm(range(200), desc = 'inner', position = 1, leave = False):
        k = str(k)
        if len(k) == 1:
          file_name = str(i)+'00'+k+'NewsData.txt'
        elif len(k) == 2:
          file_name = str(i)+'0'+k+'NewsData.txt'
        else:
          file_name = str(i)+k+'NewsData.txt'

        with open(os.path.join(args.newsData_dir,f'{i}/{file_name}'), 'r') as file:
          content = file.read()
          data.append(content)
          target.append(i)

    if shuffle == True:
      zipped_lists = list(zip(data, target))
      random.shuffle(zipped_lists)
      data, target = zip(*zipped_lists)
      data = list(data)
      target = list(target) 
    
    # save the data
    dict = {'data': data, 'target': target}
    with open(os.path.join(args.newsData_dir, f'data.pickle'), 'wb') as f:
      pickle.dump(dict, f)  

    return dict

class BERTDataset(Dataset):
  def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
    transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length = max_len, pad = pad, pair= pair)

    self.sentences = [transform([i[sent_idx]]) for i in dataset]
    self.labels = [np.int32(i[label_idx]) for i in dataset]

  def __getitem__(self, i):
    return (self.sentences[i] + (self.labels[i], ))
  
  def __len__(self):
    return (len(self.labels))

# classifier
# 목적: 인코더 위에 붙여서 레이블을 학습할 수 있도록 해줌
# Make Model


ANCHOR
class GetEmbeddings:
  def __init__(self, model):
    self.model = model
    tokenizer_bert = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer_bert, vocab)
    self.tokenizer = tok
    self.embeddings = []
    # claims or candidates
    #self.seq = seq
  
  def get_embeddings(self, embeddings):
    self.model.eval()
    self.embeddings.append(embeddings)

  def clear(self):
    self.embeddings = []

    #for cand in self.dataset:
      #tokens = self.tokenizer.
      #break
    
    #ss_dataloader = DataLoader(self.dataset, batchsize = self.batchsize, shuffle = False)


    #transform = nlp.data.BERTSentecneTransform(self.tokenizer, self.max_len, pad = True, pair = False)
    #tmp = [self.seq]
    #tokenized = transform(tmp)
    #results = self.model(torch.tensor([tokenized[0]]).to(self.device), [tokenized[1]], torch.tensor(tokenized[2]).to(self.device))
    
  

class BERTClassifier(nn.Module):
  def __init__(self, bert, hidden_size = 768, num_classes = 8, dr_rate = None, params = None):
    super(BERTClassifier, self).__init__()
    self.bert = bert
    self.dr_rate = dr_rate

    self.classifier = nn.Linear(hidden_size, num_classes)
    self.extracter = GetEmbeddings(self.bert)

    if dr_rate:
      self.dropout = nn.Dropout(p=dr_rate)

  def gen_attention_mask(self, token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
      attention_mask[i][:v] = 1
    return attention_mask.float()
  
  def forward(self, token_ids, valid_length, segment_ids):
    attention_mask = self.gen_attention_mask(token_ids, valid_length)
    sequence_output, pooler_output  = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
    
    # from here, Embeddings
    #token_vecs = sequence_output[-2][0]
    #sentence_embeddings = torch.mean(token_vecs, dim = 0)

    if __name__ != "__main__" or args.train == False:
      try:
        self.extracter.get_embeddings(pooler_output)
      except:
        self.extracter = GetEmbeddings(self.bert)
        self.extracter.get_embeddings(pooler_output)
    #print('pooler: {pooler_output}')

    #print(self.pooler)
    #pooler: sentence embedding만 따로 저장해서 실제 레이블(prediction x) 붙여넣기
    if self.dr_rate:
      out = self.dropout(pooler_output)
    #print('the model is updated')
    return self.classifier(out)



class BuildTrainModel:
  def __init__(self, args, dataset_train, dataset_test):
    # encoder
    # 필요한 이유: 모든 학습 데이터는 배치 단위로 입/출력이 정의되기 때문에, 입력 데이터 처리와
    # 레이블 처리를 동시에 출력하기 위해서 dataset 클래스를 정의하는 것
    # 이제 여기에 분류기(classifier)를 붙여야 레이블을 학습할 수 있을 것.

    # parameter setting

    # tokenizer
    tokenizer_bert = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer_bert, vocab, lower = False)
    self.tokenizer = tok

    # dataset
    self.data_train = BERTDataset(dataset_train, 0, 1, tok, args.max_len, True, False)
    self.data_test = BERTDataset(dataset_test, 0, 1, tok, args.max_len, True, False)
    self.train_dataloader = torch.utils.data.DataLoader(self.data_train, batch_size = args.batch_size, num_workers = 5, shuffle = True) #?
    self.test_dataloader = torch.utils.data.DataLoader(self.data_test, batch_size = args.batch_size, num_workers = 5, shuffle = True)

    # load pre-trained model
    if args.update == False:
      try:
        self.model = torch.load(os.path.join(args.model_dir, 'tuned_model.pt'))
      except:
        self.model = BERTClassifier(bertmodel, dr_rate = 0.5).to(args.device)
    else:
      self.model = BERTClassifier(bertmodel, dr_rate = 0.5).to(args.device)

    # optimizer, loss function
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
          ]

    self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    self.loss_fn = nn.CrossEntropyLoss()


    t_total = len(self.train_dataloader) * args.num_epochs
    warmup_step = int(t_total * args.warmup_ratio)
    self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

  def train_and_test(self, args):
    print("=====Train/Test model...=====")
    for e in range(args.num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        self.model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(self.train_dataloader)):
            self.optimizer.zero_grad()
            token_ids = token_ids.long().to(args.device)
            segment_ids = segment_ids.long().to(args.device)
            valid_length= valid_length
            label = label.long().to(args.device)
            out = self.model(token_ids, valid_length, segment_ids)
            #print(out)
            #print(label)

            #print(f'lables: {label}\n* out: {len(out)}')
            loss = self.loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            train_acc += self.calc_accuracy(out, label)

            ANCHOR
            # extract embeddings
            ## for extracting embeddings
            #poolers = []
            # labels = [] # we don't need this part anymore
            #if e == args.num_epochs -1:
            #  poolers.append(self.model.pooler)
              # labels.append(label) # we don't need this part anymore

        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
        self.model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(self.test_dataloader)):
            token_ids = token_ids.long().to(args.device)
            segment_ids = segment_ids.long().to(args.device)
            valid_length= valid_length
            label = label.long().to(args.device)
            out = self.model(token_ids, valid_length, segment_ids)
            test_acc += self.calc_accuracy(out, label)
        print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))

  def save(self, args):
    torch.save(self.model, os.path.join(args.model_dir, 'tuned_model.pt'))
    print("=====Model Saved=====")

  # 정확도 계산하는 함수
  def calc_accuracy(self, X,Y):
      max_vals, max_indices = torch.max(X, 1)
      train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
      return train_acc

  def softmax(self, vals, idx):
      valscpu = vals.cpu().detach().squeeze(0)
      a = 0
      for i in valscpu:
          a += np.exp(i)
      return ((np.exp(valscpu[idx]))/a).item() * 100

  # we'll realize this function in eval.py by calling model.pooler
  # def extract_embeddings(self, args, model, ss_dataset):
  #     #cate = ["정치","경제","사회", "생활/문화","세계","기술/IT", "연예", "스포츠"]
  #     #tmp = [seq]
  #     self.model.eval()
  #     ss_dataloader = DataLoader(ss_dataset, batchsize = args.ss_batchsize, shuffle=False)

  #     transform = nlp.data.BERTSentenceTransform(self.tokenizer, args.max_len, pad=True, pair=False)
  #     tokenized = transform(tmp)
  #     testModel_result = model(torch.tensor([tokenized[0]]).to(self.device), [tokenized[1]], torch.tensor(tokenized[2]).to(self.device))
  #     idx = testModel_result.argmax().cpu().item()
  #     return model.pooler
  #     #print("뉴스의 카테고리는:", cate[idx])
  #     #print("신뢰도는:", "{:.2f}%".format(softmax(testModel_result,idx)))

def main(args):

  # load train data
  try:
    with open(os.path.join(args.newsData_dir, 'data.pickle'), 'rb') as f:
      naver_news = pickle.load(f)
  except FileNotFoundError:
    naver_news = load_files(args, shuffle=True)

  # train_test split
  dataset_train = []
  dataset_test = []

  cnt = 0
  for data, target in zip(naver_news['data'], naver_news['target']):
    if cnt < 1120:
      dataset_train.append([data, str(target)])
    else:
      dataset_test.append([data, str(target)])

    cnt += 1

  print("=====Dataset is ready=====")

  # load model
  model = BuildTrainModel(args, dataset_train, dataset_test)
  tokenizer = model.tokenizer
  optimizer = model.optimizer
  scheduler = model.scheduler

  # trian and test model
  if args.train:
    model.train_and_test(args)

  # save the model
  if args.train or args.update:
    model.save(args)

  print("=====Model is now ready=====")
    #model.test


  # we'll realize this part in eval.py
  # # create two tensors: embeddings, targets
  # for i in range(len(poolers)):
  #   if i == 0:
  #     embeddings = poolers[0]
  #   else:
  #     embeddings = torch.cat((embeddings, poolers[i]), 0)

  # for i in range(len(labels)):
  #   if i == 0:
  #     targets = labels[0]
  #   else:
  #     targets = torch.cat((targets, labels[i]), 0)

  # naver_news_dict
  # naver_news_dict = {
  #     'data' : embeddings.detach().cpu().numpy(),
  #     'target' : targets.detach().cpu().numpy() 
  # }
  # with open('/content/drive/MyDrive/naver_news_dict.pickle', 'wb') as f:
  #     pickle.dump(naver_news_dict, f)

if __name__ == "__main__":
  print("=====Tuning KoBert=====")
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--newsData_dir",
                      default = './new_ss/newsData',
                      help = "Where original news data is stored")
  parser.add_argument("--max_len", 
                      default=64,
                      help="Max length for tokens")
  parser.add_argument("--batch_size",
                      default=64,
                      help = "Batch size for training")
  parser.add_argument("--warmup_ratio",
                      default=0.1,
                      help = "This prevents early-over fitting")
  parser.add_argument("--num_epochs",
                      default=10,
                      help="The number of iterations")
  parser.add_argument("--max_grad_norm",
                      default=1,
                      help="Limit the size of L2 norm")
  parser.add_argument("--log_interval",
                      default=200,
                      help="")
  parser.add_argument("--learning_rate",
                      default=5e-5,
                      help="")
  parser.add_argument("--model_dir",
                      default="./new_ss/",
                      help = "Where the fine-tuned model will be stored")
  parser.add_argument("--train",
                      default=False,
                      action='store_true',
                      help = "Train the model")
  parser.add_argument("--update",
                      default = False,
                      action = 'store_true', 
                      help = "Update the model")

  args = parser.parse_args()
  args.device = "cuda" if torch.cuda.is_available() else "cpu"

  main(args)
