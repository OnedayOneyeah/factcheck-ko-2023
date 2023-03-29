import os
import re
import json
import pickle
import random
import argparse
import logging
import socket
from collections import defaultdict
from unittest.case import DIFF_OMITTED
import numpy as np
from tqdm import tqdm
import datetime
import torch
from torch.nn import functional as F
from torch.utils.data import (
    TensorDataset, 
    DataLoader,
    RandomSampler,
)
from torch.nn.utils import clip_grad_norm_

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

# for multi-gpu training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import warnings
from transformers import logging as trans_logging
warnings.filterwarnings("ignore")
trans_logging.set_verbosity_error()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cleanse_and_split(text, rm_parens=False):
    while re.search(r"\[[^\.\[\]]*\]", text):
        text = re.sub(r"\[[^\.\[\]]*\]", "", text)
    text = re.sub(r" 이 소리의 정보[^,\)]*\(도움말·정보\)", "", text)
    text = re.sub(r"(\([^\(\)]*)[\s\,]+(듣기|울음소리)[\s\,]+([^\(\)]*\))", r"\g<1>, \g<3>", text)

    if rm_parens:
        while re.search(r"\([^\(\)]*\)", text):
            text = re.sub(r"\([^\(\)]*\)", "", text)
    else:
        while True:
            lang_search = re.search(r"\([^\(\)]*(:)[^\(\)]*\)", text)  # e.g. 영어: / 프랑스어: / 문화어: ... => Delete
            if lang_search:
                lang_start, lang_end = lang_search.span()
                lang_replace = re.sub(r"(?<=[\(\,])[^\,]*:[^\,]*(?=[\,\)])", "", lang_search.group())
                if lang_replace == lang_search.group():
                    logger.warning(f"An unexpected pattern showed up! {text}")
                text = text[:lang_start] + lang_replace + text[lang_end:]
            else:
                break

        text = re.sub("\([\s\,]*\)", "", text)
        while re.search(r"\([^\.\(\)]*\)", text):
            text = re.sub(r"\(([^\.\(\)]*)\)", r" \g<1> ", text)

    text = re.sub(r"·", " ", text)
    text = re.sub(r"<+([^\.<>]*)>+", r"\g<1>", text)
    text = re.sub(r"《+([^\.《》]*)》+", r"\g<1>", text)

    text = re.sub(r"=+.*?=+", "", text)

    text = re.sub(r"[^가-힣a-zA-Z0-9\s\.\?]+", "", text)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"[\n]+", "\n", text)

    # truncate_from = re.search(r"(같이 보기|각주|참고문헌).*편집]", text)
    # if truncate_from:
    #     text = text[:truncate_from.start()]

    test_lst = [t2.strip() for t in text.split("\n") for t2 in re.split(r'[\.\?][\s]|[\.\?]$', t.strip()) if len(t2) >= 10]
    return test_lst


def add_noise(args):
    with open(os.path.join(args.input_dir, "wiki_claims.json"), "r") as fp:
        claims = json.load(fp)
    with open(os.path.join(args.dr_dir, "dr_results.json"), "r") as fp:
        dr_results = json.load(fp)
    with open(os.path.join(args.corpus_dir, "wiki_docs.json"), "r") as fp:
        wiki = json.load(fp)
        wiki_titles = wiki.keys()

    noise_dict = defaultdict(dict) # {cid: [candidates]}
    warnings = defaultdict(dict)
    
    for cid in claims:
        data = claims[cid]
        titles_annotated = list(set([data[f"title{i}"] for i in range(1, 6) if data[f"title{i}"]]))
        # if len(titles_annotated) == 0:
        #         logger.warning(f"claim id {cid} ... No title is annotated. This claim will be Dropped!")
        #         warnings[cid]["No title"] = []
        #         continue
        existing_titles = [title for title in list(set(dr_results[cid] + titles_annotated)) if title in wiki_titles] 
            # 클레임에 해당하는 document retrieval results 위키 문서들의 타이틀, 혹은 클레임 데이터 자체에 있는 타이틀(1~5)이 wiki_titles 문서 안에 있다면
            # existing_titles로 포함시켜라.
        
        candidates = []
        for title in existing_titles:
            documents = wiki[title]
            date = datetime.datetime.strptime(data["Date"], "%Y-%m-%d %H:%M:%S.%f")
            doc_dates = [datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%SZ") for dt in documents.keys()]
            doc_dates = [dt for dt in doc_dates if dt <= date]
            if not doc_dates:
                warnings[cid]["Wrong Date"] = {"Date annotated": data["Date"],
                                               "Dates in downloaded wiki documents": [d.strftime("%Y-%m-%d %H:%M:%S.%f")
                                                                                      for d in doc_dates]}
                continue
            text = documents[max(doc_dates).strftime("%Y-%m-%dT%H:%M:%SZ")] # 가장 최신 문서로 가져와라
            text_lst = cleanse_and_split(text) # str cleaning 이후, 길이 10인 문장들만 뽑아서 가져옴
            candidates.extend(text_lst)
        noise = random.sample(candidates, 1)
        assert type(noise) == list, candidates

        noise_dict[cid] = noise
    
    return noise_dict

    # input: cid / output: a sentence from the retrieved doc


def get_dataset(args, split):
    make_noise = args.add_noise
    logger.info(f"Make {split} dataset")

    with open(os.path.join(args.input_dir, "train_val_test_ids.json"), "r") as fp:
        split_ids = json.load(fp)[f"{split}_ids"]

    with open(os.path.join(args.input_dir, "wiki_claims.json"), "r") as fp:
        claims = json.load(fp)
        claims = {cid: data for cid, data in claims.items() if cid in split_ids}
        if args.debug:
            claims = {cid: data for i, (cid, data) in enumerate(claims.items()) if i < 100}

    with open(os.path.join(args.ss_dir, "nei_ss_results.json"), "r") as fp:
        nei_ss_results = json.load(fp)
        nei_ss_results = {cid: result for cid, result in nei_ss_results.items() if cid in split_ids}

    # add noise #
    noise_dict = add_noise(args) if make_noise else None
    # ========= #
    examples = []
    warnings = defaultdict(dict)
    

    if args.v1_rc == 'cls':
        name2label = {'True': 0, 'False': 1, 'None': 2}
    else:
        name2label = {'True': 1.0, 'False': -1.0, 'None': 0.0}

    for cid in claims:
        data = claims[cid]
        claim = data['claim']
        label = name2label[data['True_False']]

        if (args.v1_rc == 'cls' and label != 2) or (args.v1_rc == 'rgs' and label != 0.0):
            evidences_anno = [data[f"evidence{i}"] for i in range(1, 6) if data[f"evidence{i}"]]
            # 여기서 evidence는 human-annotated된 근거 자료들을 의미함
            evidences_clean = [ev_clean for ev_anno in evidences_anno
                               for ev_clean in cleanse_and_split(ev_anno) if len(ev_clean) >= 10]
            if len(evidences_clean) == 0:
                warnings[cid]["No evidences annotated but not NEI"] = [claim]
                continue

        else:  # NEI -> use ss results (ss/dir_name/nei_ss_results.json)
            evidences_clean = nei_ss_results[cid][:2]

        # add noise
        if make_noise:
            noise = noise_dict[cid]
            #print(noise)
            evidences_clean = evidences_clean + noise
            random.shuffle(evidences_clean)

        examples.append({
            'claim': claim,
            'evidences': evidences_clean,
            'label': label,
        })

    logger.info(f"# of claims in {split} dataset: {len(claims)}")
    logger.info(f"# of claims that is something wrong: {len(warnings)} / {len(claims)}")
    logger.info(f"# of claim-evidence pairs in {split} dataset: {len(examples)}")
    if not args.debug:
        with open(f"{args.input_dir}/{split}_rte_warnings.json", "w") as fp:
            json.dump(warnings, fp, indent=4, ensure_ascii=False)

    return examples


def convert_dataset(args, examples, tokenizer, max_length, split, predict=False, display_examples=False):
    """
    Convert train examples into BERT's input foramt.
    """
    features = []
    for ex_idx, example in tqdm(enumerate(examples), total=len(examples)):
        sentence_b = tokenizer.tokenize(example['claim'])
        # if split != "train":  # "val" or "test"
        #     per_claim_features = []

        if not predict:
            label = example['label']

        if split != "train" and args.v2_dataset == "mpe":
            iteration = 1
        else:
            if args.ev_num >= len(example['evidences']):
                iteration = 1
            else:
                iteration = len(example['evidences']) - args.ev_num + 1

        for idx in range(iteration):
            if split != "train" and args.v2_dataset == "mpe":
                ev = random.sample(example['evidences'], k = min(args.ev_num, len(example['evidences'])))
            else:
                ev = example['evidences'][idx:idx+args.ev_num] # aggregated evidence sentences
            sentence_a = [tokenizer.tokenize(x) for x in ev]
            sen_lens = [len(sentence_a[i])+1 for i in range(len(sentence_a))]
            sentence_a = [y for i, x in enumerate(sentence_a) for y in (x if i==0 else ['[SEP]'] +x)]

            # ev = example['evidences'][idx]
            #sentence_a = tokenizer.tokenize(ev)

            if len(sentence_a) + len(sentence_b) > max_length - 3 - (args.ev_num):  # 3 for [CLS], 2x[SEP]
                logger.warning(
                    "The length of the input is longer than max_length! "
                    f"sentence_a: {sentence_a} / sentence_b: {sentence_b}"
                )
                # truncate sentence_b to fit in max_length
                diff = (len(sentence_a) + len(sentence_b)) - (max_length - 3 - (args.ev_num))
                if args.v2_dataset== 'mpe':
                    ptr = 1
                    # locate
                    while sum(sen_lens[-ptr:]) < diff:
                        ptr += 1
                    # pop, if necessary
                    val = sum(sen_lens[-ptr:]) - diff
                    sen_lens = sen_lens[::-(ptr-1)] if ptr >1 else sen_lens
                    sen_lens[-1] = val
                sentence_a = sentence_a[:-diff]

            tokens = ["[CLS]"] + sentence_a + ["[SEP]"] + sentence_b + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # segment_ids = [0] * (len(sentence_a) + 2) + [1] * (len(sentence_b) + 1)
            if args.v2_dataset == 'mpe':
                segment_ids = [0 if i%2 == 0 else 1 for i,x in enumerate(sen_lens) for y in range(x)] +[(len(sen_lens)+1)%2] + [len(sen_lens)%2]*(len(sentence_b)+1) 
            else:
                segment_ids = [0] * (len(sentence_a)+2) + [1] * (len(sentence_b) + 1)
            mask = [1] * len(input_ids)

            # Zero-padding
            padding = [0] * (max_length - len(input_ids))
            input_ids += padding
            segment_ids += padding
            mask += padding
            assert len(input_ids) == max_length
            assert len(segment_ids) == max_length, f'{len(segment_ids)},{max_length}'
            assert len(mask) == max_length, f'{len(mask)}, {max_length}'

            if ex_idx < 3 and display_examples:
                logger.info(f"========= Train Example {ex_idx + 1} =========")
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("mask: %s" % " ".join([str(x) for x in mask]))
                logger.info("label: %s" % label)
                logger.info("")

            if not predict:
                feat = {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': mask, 'label': label}
                # if split == "train":
                #     features.append(feat)
                # else:
                #     per_claim_features.append(feat)
                features.append(feat)
            else:  # predict
                feat = {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': mask}
                # per_claim_features.append(feat)
                features.append(feat)

        # if split != "train":
        #     features.append(per_claim_features)

    return features


def load_or_make_features(args, tokenizer, split, save=True):
    small = '_small' if args.debug else ''
    rgs_cls = '_'+args.v1_rc
    spe_mpe = '_'+args.v2_dataset
    exclude_nei = '_nei_excluded' if args.v3_exclude_nei else ''
    ev_num = '_'+str(args.ev_num) if args.v2_dataset == "mpe" else ''
    add_noise = '_noised' if args.add_noise else ''
    features_path = os.path.join(args.temp_dir, f"{split}_features{small}{rgs_cls}{spe_mpe}{exclude_nei}{ev_num}{add_noise}.pickle")

    try:
        with open(features_path, "rb") as fp:
            features = pickle.load(fp)
        return features

    except FileNotFoundError or EOFError:
        dataset = get_dataset(args, split=split)
        features = convert_dataset(args, dataset, tokenizer, args.max_length, split=split)
        if save:
            with open(features_path, "wb") as fp:
                pickle.dump(features, fp)
        return features


def build_rte_model(args, num_labels=3):
    if args.model == "koelectra":
        return ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator",
                                                                cache_dir=args.cache_dir, num_labels=num_labels)
    else:
        return BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',
                                                             cache_dir=args.cache_dir, num_labels=num_labels)


def main_worker(gpu, train_dataset, val_features, args):
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:9856", rank=args.gpu, world_size=args.n_gpu)

    if args.v1_rc == 'rgs':
        model = build_rte_model(args, num_labels=1)
    else:
        model = build_rte_model(args, num_labels=3)

    model = model.to(args.gpu)
    model = DDP(model, device_ids=[args.gpu])

    if args.evaluate:
        dist.barrier()
        rgs_cls = '_' + args.v1_rc
        spe_mpe = '_' + args.v2_dataset
        exclude_nei = '_nei_excluded' if args.v3_exclude_nei else ''
        ev_num = '_'+str(args.ev_num) if args.v2_dataset == "mpe" else ''
        add_noise = '_noised' if args.add_noise else ''
        ckpt_path = os.path.join(args.checkpoints_dir, f"best_ckpt{rgs_cls}{spe_mpe}{exclude_nei}{ev_num}{add_noise}.pth")
        checkpoint = torch.load(ckpt_path, map_location=f"cuda:{args.gpu}")
        model.module.load_state_dict(checkpoint["state_dict"])
        validate(val_features, model, None, None, args)
        return None

    # prepare optimizer
    num_train_optimization_steps = int(len(train_dataset) / args.batchsize) * args.num_train_epochs \
        if train_dataset else -1
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_optimization_steps * args.warmup_proportion),
        num_training_steps=num_train_optimization_steps
    )

    if args.multiproc_dist:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batchsize,
    )

    dist.barrier()

    # TRAINING !!
    if args.gpu == 0:
        print("=== Running training ===")
        print("===== Num examples : %d" % len(train_dataset))
        print("===== Batch size : %d" % args.batchsize)
        print("===== Num steps : %d"% num_train_optimization_steps)

    epoch_start = 0
    args.best_acc = 0
    if not args.debug:
        rgs_cls = '_' + args.v1_rc
        spe_mpe = '_' + args.v2_dataset
        exclude_nei = '_nei_excluded' if args.v3_exclude_nei else ''
        ev_num = '_'+str(args.ev_num) if args.v2_dataset == "mpe" else ''
        add_noise = '_noised' if args.add_noise else ''
        ckpt_path = os.path.join(args.checkpoints_dir, f"best_ckpt{rgs_cls}{spe_mpe}{exclude_nei}{ev_num}{add_noise}.pth")
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=f"cuda:{args.gpu}")
            epoch_start = checkpoint["epoch"]
            model.module.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            args.best_acc = checkpoint["best_acc"]
            if args.gpu == 0:
                print(f"=> Loading checkpoint {ckpt_path} to cuda:{args.gpu}, epoch from {epoch_start}")

    for epoch in range(epoch_start, args.num_train_epochs):
        if args.gpu == 0:
            print(f"\n===== Epoch {epoch + 1} =====")
        train_sampler.set_epoch(epoch)
        args.epoch = epoch

        train(train_dataloader, model, optimizer, scheduler, args)
        dist.barrier()

        validate(val_features, model, optimizer, scheduler, args)


def train(train_dataloader, model, optimizer, scheduler, args):
    model.train()

    if args.gpu == 0:
        tr_loss, num_tr_steps = 0, 0
        temp_tr_loss, temp_num_tr_steps = 0, 0
        pbar = tqdm(total=len(train_dataloader), desc="Iteration")

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(args.gpu) for t in batch)
        input_ids, segment_ids, input_mask, labels = batch
        model.zero_grad()
        outputs = model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=labels
        )
        loss, logits = outputs.loss, outputs.logits

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if args.gpu == 0:
            tr_loss += loss.item()
            num_tr_steps += 1

            # logging every 20% of total steps
            temp_tr_loss += loss.item()
            temp_num_tr_steps += 1
            log_every = int(len(train_dataloader) * 0.2) if len(train_dataloader) * 0.2 >= 1 else 1
            if (step + 1) % log_every == 0:
                print(
                    "Epoch %d/%d - step %d/%d" % ((args.epoch + 1), args.num_train_epochs, step, len(train_dataloader)))
                print("Logging from cuda: %d" % args.gpu)
                print("temp loss %f" % (temp_tr_loss / temp_num_tr_steps))
                temp_tr_loss, temp_num_tr_steps = 0, 0

            pbar.update(1)

    if args.gpu == 0:
        print("===== Training Done =====")


def validate(val_features, model, optimizer, scheduler, args):

    all_input_ids = torch.LongTensor([x['input_ids'] for x in val_features])
    all_segment_ids = torch.LongTensor([x['segment_ids'] for x in val_features])
    all_input_masks = torch.LongTensor([x['input_mask'] for x in val_features])
    if args.v1_rc == 'cls':
        all_label = torch.LongTensor([x['label'] for x in val_features])
    else:
        all_label = torch.FloatTensor([x['label'] for x in val_features])
    val_dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_masks, all_label)
    val_sampler = DistributedSampler(val_dataset, shuffle = True)

    val_dataloader = DataLoader(
        val_dataset,
        sampler = val_sampler,
        batch_size = args.batchsize
    )

    model.eval()
    val_loss, val_acc = 0, 0
    n_claims_each_label = torch.LongTensor([0, 0, 0])  # rgs: Supported, Refuted, NEI
    n_corrects_each_label = torch.LongTensor([0, 0, 0]) # cls: Refuted, NEI, Supported
    n_predicted_each_label = torch.LongTensor([0, 0, 0])

    if args.gpu == 0:
        print("=== Running validation ===")
        pbar = tqdm(total=len(val_dataloader), desc="Iteration")
    
    doc_losses, doc_logits = [], []
    doc_labels_cat = []

    cnt = 0
    for batch in val_dataloader:
        batch = tuple(t.to(args.gpu) for t in batch)
        input_ids, segment_ids, input_mask, labels = batch

        with torch.no_grad():
            outputs = model(
                input_ids,
                token_type_ids = segment_ids,
                attention_mask = input_mask,
                labels = labels
            )
        
        if args.v1_rc == 'cls':
            loss, logits = outputs.loss, outputs.logits
            doc_losses.append(loss)
            doc_logits.append(logits)

        else:
            loss, logits = outputs.loss, outputs.logits.transpose(1,0)[0]
            doc_losses.append(loss.item())
            doc_logits.extend(logits)

        if args.gpu == 0:
            pbar.update(1)

        doc_labels_cat.extend(labels)

        val_loss += loss
        cnt += 1

    #print(doc_logits)
    #assert 1 == 2
    doc_labels_cat = np.array([int(x.item()) for x in doc_labels_cat])
    is_correct, pred = calculate_metric(args, doc_logits, doc_labels_cat)
    
    # for check
    with open(os.path.join(os.getcwd(), 'new_rte/tmp/is_correct.pickle'), 'wb') as f:
        pickle.dump(is_correct, f)
    with open(os.path.join(os.getcwd(), 'new_rte/tmp/pred.pickle'), 'wb') as f:
        pickle.dump(pred, f)

    val_loss /= cnt
    val_acc = len([1 for x in is_correct if x == 1])
    val_acc /= len(is_correct)

    # summary
    for c,l,p in zip(is_correct, doc_labels_cat, pred):
        if args.v1_rc == 'rgs':
            l += 1
            p += 1
        n_corrects_each_label[l] += c # claim 레이블 각각 몇 개 맞췄는지 [0,1,2]
        n_claims_each_label[l] += 1 # claim 레이블 합쳐서 각각 몇 개인지 [0,1,2]
        n_predicted_each_label[p] += 1 # 예측한 claim 레이블 각각 몇 개인지 [0,1,2]


    val_results = torch.tensor([val_loss, val_acc]).to(args.gpu)
    n_claims_each_label = n_claims_each_label.to(args.gpu)
    n_corrects_each_label = n_corrects_each_label.to(args.gpu)
    n_predicted_each_label = n_predicted_each_label.to(args.gpu)
    dist.barrier()
    dist.reduce(val_results, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(n_claims_each_label, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(n_corrects_each_label, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(n_predicted_each_label, dst=0, op=dist.ReduceOp.SUM)


    if args.gpu == 0:
        val_results /= args.n_gpu
        val_loss, val_acc = tuple(val_results.tolist())

        print("===== Validation loss", val_loss)
        print("===== Validation accuracy", val_acc)

        recall_each_label = n_corrects_each_label / n_claims_each_label
        precision_each_label = n_corrects_each_label / n_predicted_each_label

        # name2label = {'False': 0.0, 'None': 1.0, 'True': 2.0}
        rgs_label = ['Refuted', 'NEI', 'Supported']
        cls_label = ['Supported', 'Refuted', 'NEI']

        if args.v1_rc == 'rgs':
            labels = rgs_label
        else:
            labels = cls_label

        print('===== Recall for each Labels =====')
        for i, l in enumerate(labels):
            print(f'label{i} ({l}): {recall_each_label[i]} ({n_corrects_each_label[i]}/{n_claims_each_label[i]})')
        print('===== Precision for each Labels =====')
        for i, l in enumerate(labels):
            print(f'label{i} ({l}) : {precision_each_label[i]} ({n_corrects_each_label[i]}/{n_predicted_each_label[i]})')

        if not args.evaluate and val_acc > args.best_acc:
            args.best_acc = val_acc
            rgs_cls = '_' + args.v1_rc
            spe_mpe = '_' + args.v2_dataset
            exclude_nei = '_nei_excluded' if args.v3_exclude_nei else ''
            ev_num = '_'+str(args.ev_num) if args.v2_dataset == "mpe" else ''
            add_noise = '_noised' if args.add_noise else ''
            path = os.path.join(args.checkpoints_dir, f"best_ckpt{rgs_cls}{spe_mpe}{exclude_nei}{ev_num}{add_noise}.pth")

            if not args.debug:
                print(f'===== New Accuracy record! save checkpoint (epoch: {args.epoch + 1})')
                torch.save({
                    'epoch': args.epoch + 1,
                    'best_acc': args.best_acc,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, path)


def calculate_metric(args, score_or_logits, doc_labels_cat):
    #mode = args.mode # we don't need this anymore

    if args.v1_rc == 'rgs':
        score_or_logits = np.array([x.item() for x in score_or_logits])
        sf, st = np.quantile(score_or_logits, 0.5-args.nei_scale*0.5), np.quantile(score_or_logits, 0.5+args.nei_scale*0.5)

    else: #'cls'
        score_or_logits = torch.cat(score_or_logits, dim = 0)
        softmax_logits = F.softmax(score_or_logits, dim = 1)
        pred_for_each_claim = softmax_logits.argmax(dim  = 1)
        score_or_logits = np.array([x.item() for x in pred_for_each_claim])

    assert len(score_or_logits) == len(doc_labels_cat)
    is_correct = []
    pred = []
    
    for l, p in zip(doc_labels_cat, score_or_logits):
        if args.v1_rc == 'rgs':
            if p < sf:
                p = -1 # Refuted
            elif p >= sf and p < st:
                p = 0 # NEI
            else:
                p = 1 # Supported

        pred.append(p)

        if int(l) == int(p):
            is_correct.append(1)
        else:
            is_correct.append(0)
    
    return is_correct, pred

def main():
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument("--input_dir",
                        default="./data/",
                        type=str,
                        help="The input data dir.")
    parser.add_argument("--dr_dir",
                        default="./dr",
                        type=str,
                        help="The results of document retrieval dir.")
    parser.add_argument("--corpus_dir",
                        default="./data/wiki/",
                        type=str,
                        help="The wikipedia corpus dir.")
    parser.add_argument("--ss_dir",
                        default="./ss/",
                        type=str,
                        help="To get the ss results of NEI data.")
    parser.add_argument("--temp_dir",
                        default="./new_rte/tmp/",
                        type=str,
                        help="The temp dir where the processed data file will be saved.")
    parser.add_argument("--checkpoints_dir",
                        default="./new_rte/checkpoints/",
                        type=str,
                        help="Where checkpoints will be stored.")
    parser.add_argument("--cache_dir",
                        default="./data/models/",
                        type=str,
                        help="Where do you want to store the pre-trained models"
                        "downloaded from pytorch pretrained model.")
    parser.add_argument("--batchsize",
                        default=8,
                        type=int,
                        help="Batch size for training examples.")
    parser.add_argument("--learning_rate",
                        default=1e-6,
                        type=float,
                        help="The initial learning rate.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs.")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="random seed.")
    parser.add_argument("--max_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after tokenized."
                        "If longer than this, it will be truncated, else will be padded.")
    parser.add_argument('--multiproc_dist',
                        default=False,
                        action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs.')
    parser.add_argument('--debug',
                        default=False,
                        action='store_true',
                        help='Use small datasets to debug.')
    parser.add_argument("--evaluate",
                        default=False,
                        action='store_true',
                        help="Validation only (No training).")
    parser.add_argument('--test',
                        default=False,
                        action='store_true',
                        help='Use test dataset to evaluate.')
    parser.add_argument('--model',
                        default="koelectra",
                        type=str,
                        help='"koelectra" if want to use KoElectra model (https://github.com/monologg/KoELECTRA).')
    parser.add_argument('--mode',
                        default = "sum",
                        type = str,
                        help = "calculate metric mode")
    parser.add_argument('--v1_rc',
                        default = 'cls',
                        type = str,
                        help = 'Regression(rgs) or Classification(mcls, bcls)')
    parser.add_argument('--v2_dataset',
                        default = 'mpe',
                        type = str,
                        help = 'Single Premise Entailment(spe) or Multiple Premises Entailment(mpe)')
    parser.add_argument('--v3_exclude_nei',
                        default = False,
                        action = 'store_true',
                        help = "Exclude nei data from train/val/test dataset")
    parser.add_argument('--nei_scale',
                        default = 0.45,
                        type = float,
                        help = "Determines the range of the predictions to be labeled as 'nei' in the regression model")
    parser.add_argument('--ev_num',
                        default = 4,
                        type = int,
                        help = 'The number of evidence sentences integrated with a claim when building datasets')
    parser.add_argument('--add_noise',
                        default = False,
                        action = 'store_true',
                        help = "include retrieved sentences in a train and validate processes" )
    args = parser.parse_args()

    if args.multiproc_dist:
        args.n_gpu = torch.cuda.device_count()
        args.gpu = None
    else:
        args.n_gpu = 1
        args.gpu = 0
    
    assert args.v1_rc in ['cls', 'rgs']
    assert args.v2_dataset in ['mpe', 'spe']

    if args.v2_dataset == 'spe':
        args.ev_num = 1
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG)

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Make sure to pass do_lower_case=False when use multilingual-cased model.
    # See https://github.com/google-research/bert/blob/master/multilingual.md
    if args.model == "koelectra":
        tokenizer = ElectraTokenizerFast.from_pretrained("monologg/koelectra-base-v3-discriminator")
    else:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    if args.evaluate:
        val_split = "test" if args.test else "val"
        val_features = load_or_make_features(args, tokenizer, split=val_split)
        train_dataset = None

    else:
        train_features = load_or_make_features(args, tokenizer, split="train")
        val_features = load_or_make_features(args, tokenizer, split="val")

        # TensorDataset, sampler
        all_input_ids = torch.LongTensor([x['input_ids'] for x in train_features])
        all_segment_ids = torch.LongTensor([x['segment_ids'] for x in train_features])
        all_input_masks = torch.LongTensor([x['input_mask'] for x in train_features])
        if args.v1_rc == 'cls':
            all_label = torch.LongTensor([x['label'] for x in train_features])
        else:
            all_label = torch.FloatTensor([x['label'] for x in train_features])
        train_dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_masks, all_label)

    logger.info("===== Data is prepared =====")

    if args.multiproc_dist:
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=args.n_gpu,
                 args=(train_dataset, val_features, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, train_dataset, val_features, args)


if __name__ == "__main__":
    print(f"Job is running on {socket.gethostname()}")
    main()

