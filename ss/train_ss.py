import os
import re
import json
import random
import pickle
import argparse
import logging
import socket
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
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


def matching_scorer(evidence, candidate):
    evidence_unigrams = evidence.split()
    candidate_unigrams = candidate.split()
    # evidence_bigrams = [" ".join(evidence_unigrams[i:i+2]) for i in range(len(evidence_unigrams) - 1)]
    # candidate_bigrams = [" ".join(candidate_unigrams[i:i+2]) for i in range(len(candidate_unigrams) - 1)]
    inner_join = (set(evidence_unigrams) & set(candidate_unigrams))
    # outer_join = (set(evidence_unigrams) | set(candidate_unigrams))
    ev_score = len(inner_join) / len(set(evidence_unigrams))
    cand_score = len(inner_join) / len(set(candidate_unigrams))
    return (ev_score + cand_score) / 2


def labeler(ev, candidates, labels, thres=0.75):
    matched = False
    ev = ev[:-1] if ev[-1] == "." else ev  # Handling double periods (~..)
    for i, cand in enumerate(candidates):
        score = matching_scorer(ev, cand)
        if "." in cand:
            cand_lst = [c for c in cand.split(".") if len(c) >= 10]
            if cand_lst:
                cand_scores = [matching_scorer(ev, c) for c in cand_lst]
                cand_labels = [0] * len(cand_lst)
                cand_max_score = max(cand_scores)
                if (cand_max_score > score) and (cand_max_score > thres):
                    cand_labels[cand_scores.index(max(cand_scores))] = 1
                    del candidates[i], labels[i]
                    candidates.extend(cand_lst)
                    labels.extend(cand_labels)
                    matched = True
                    continue
        if score >= thres:  # Note: more than on candidates can be matched for one evidence
            labels[candidates.index(cand)] = 1
            matched = True

    return matched, candidates, labels


def get_data(args, split):
    print(f"Make {split} data")

    with open(os.path.join(args.input_dir, "train_val_test_ids.json"), "r") as fp:
        split_ids = json.load(fp)[f"{split}_ids"]

    with open(os.path.join(args.input_dir, "wiki_claims.json"), "r") as fp:
        claims = json.load(fp)
        # nei samples should be dropped because they essentially don't contain gold 'evidence'
        claims = {cid: data for cid, data in claims.items() if cid in split_ids and data["True_False"] != "None"}
        # cid: claim id, data: corresponding data dict {user id, evidence1, ...}
        if args.debug:
            claims = {cid: data for i, (cid, data) in enumerate(claims.items()) if i < 500}

    with open(os.path.join(args.dr_dir, "dr_results.json"), "r") as fp:
        dr_results = json.load(fp)
        dr_results = {cid: dr_result for cid, dr_result in dr_results.items() if cid in split_ids} #dr_result: titles
        # dr_results / wiki docs - 전자는 {id: [wiki_titles]}, 후자는 {title: {datetime: docs}}

    with open(os.path.join(args.corpus_dir, "wiki_docs.json"), "r") as fp:
        wiki = json.load(fp) #{title: } # Wikipedia documents corresponing to claims in wiki_claims.json
        wiki_titles = wiki.keys() # titles of each doc corresponding to each claim

    data_list = []
    warnings = defaultdict(dict)  # "id": {"warning message": some info, ...}
    for cid in claims:
        data = claims[cid] 
        titles_annotated = list(set([data[f"title{i}"] for i in range(1, 6) if data[f"title{i}"]]))
        # claim 데이터에 타이틀이 있으면, 그 타이틀 유니크한 값들을 가져와서 리스트로 만들어라.
        if len(titles_annotated) == 0:
            logger.warning(f"claim id {cid} ... No title is annotated. This claim will be Dropped!")
            warnings[cid]["No title"] = []
            continue
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

        labels = [0] * len(candidates)

        # ----- Handle evidences ----- #
        evidences = [data[f"evidence{i}"] for i in range(1, 6) if data[f"evidence{i}"]]
        # claim에 해당하는 evidence가 있다면, 리스트 안에 집어넣어라!

        save = True  # Do not save this claim if we can't find the first evidence from candidates
        for i, ev_anno in enumerate(evidences):
            ev_lst = [e for e in cleanse_and_split(ev_anno) if len(e) >= 10]
            # ev = ev_lst[max(range(len(ev_lst)), key=lambda i: len(ev_lst[i]))]  # Take the longest part of the list
            for ev in ev_lst:
                ev_matched, candidates, labels = labeler(ev, candidates, labels)
                if not ev_matched:
                    if "." in ev:  # Handle Special Case: periods in the middle of the evidence sentence
                        matched_bools = []
                        unmatched_lst = []
                        ev2_lst = [e for e in ev.split(".") if len(e) >= 10]
                        for ev2 in ev2_lst:
                            ev2_matched, candidates, labels = labeler(ev2, candidates, labels)
                            matched_bools.append(ev2_matched)
                            if not ev2_matched:
                                unmatched_lst.append(ev2)
                        if sum(matched_bools) > 0:  # any match, append only un-matched ev2
                            if unmatched_lst:
                                logger.warning(
                                    f"claim id {cid} ... Can't find evidence {unmatched_lst} from the candidates.")
                                warnings[cid]["wrong_evidences"] = unmatched_lst
                        else:  # no match
                            logger.warning(f"claim id {cid} ... Can't find evidence [{ev[:20]}] from the candidates.")
                            warnings[cid]["wrong_evidences"] = ev
                    else:
                        logger.warning(f"claim id {cid} ... Can't find evidence [{ev[:20]}] from the candidates.")
                        warnings[cid]["wrong_evidences"] = ev

            if i == 0 and sum(labels) == 0:
                save = False
                logger.warning(f"claim id {cid} ... Can't find the first evidence [{ev_anno[:20]}] from the candidates")
                warnings[cid]["wrong_evidences"] = ev_anno
                break

        if save:
            claim = data['claim']
            data_list.append({
                'id': cid,
                'claim': claim,
                'candidates': candidates,
                'labels': labels,
                'more_than_two': data["more_than_two"]
            })

    total_n_sentences = 0
    total_ratio = 0
    for d in data_list:
        total_n_sentences += len(d["candidates"])
        total_ratio += sum(d["labels"]) / len(d["labels"]) # true 값 비율

    print(f"<<{split} set ss labelling results>>")
    print(f"# claims that have wrong titles or evidences: {len(warnings)} / {len(claims)}")
    print(f"average # sentences: {total_n_sentences / len(data_list)}")
    print(f"average evidence sentence ratio: {total_ratio / len(data_list)}")
    if not args.debug:
        with open(f"{args.input_dir}/{split}_ss_warnings.json", "w") as fp:
            json.dump(warnings, fp, indent=4, ensure_ascii=False)

    print(f"# claims left: {len(data_list)} / {len(claims)}")
    return data_list


def load_or_make_data_chunks(args, split, save=True):
    small = '_small' if args.debug else ''
    data_path = os.path.join(args.temp_dir, f"{split}_data{small}.pickle")

    try:
        with open(data_path, "rb") as fp:
            data = pickle.load(fp)
    except FileNotFoundError or EOFError:
        data = get_data(args, split=split)
        if save:
            with open(data_path, "wb") as fp:
                pickle.dump(data, fp)

    if split == "train":  # return data chunks
        chunksize = int(len(data) / args.num_chunks) + 1 # default: 10
        return [data[i: i + chunksize] for i in range(0, len(data), chunksize)] # 2차원 리스트. 미니 배치 만드는 것처럼, observation 하나당 10개씩
    else:
        return data


def convert_bert_features(args, examples, tokenizer, split, predict=False, display_examples=False):
    """
    Convert train examples into BERT's input foramt.
    """
    if split == "train":
        features_pos = []
        features_neg = []
    else:  # "val" or "test"
        val_features = []

    for ex_idx, example in tqdm(enumerate(examples), total=len(examples)): #examples: train_data
        sentence_b = tokenizer.tokenize(example['claim']) #bert tokenizer
        if split != "train":
            per_claim_features = []

        for idx in range(len(example['candidates'])):
            cand = example['candidates'][idx]
            if not predict:
                label = example['labels'][idx]
            sentence_a = tokenizer.tokenize(cand)

            if len(sentence_a) + len(sentence_b) > args.max_length - 3:  # 3 for [CLS], 2x[SEP]
                # logger.warning(
                #     "The length of the input is longer than max_length! "
                #     f"sentence_a: {sentence_a} / sentence_b: {sentence_b}"
                # )
                # truncate sentence_a to fit in max_length
                diff = (len(sentence_a) + len(sentence_b)) - (args.max_length - 3)
                sentence_a = sentence_a[:-diff]

            tokens = ["[CLS]"] + sentence_a + ["[SEP]"] + sentence_b + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * (len(sentence_a) + 2) + [1] * (len(sentence_b) + 1)
            # to distinguish two sentences: [0] for first sentence, [1] for second one (including [CLS] and [SEP])
            mask = [1] * len(input_ids)

            # Zero-padding
            padding = [0] * (args.max_length - len(input_ids))
            input_ids += padding
            segment_ids += padding
            mask += padding
            assert len(input_ids) == args.max_length
            assert len(segment_ids) == args.max_length
            assert len(mask) == args.max_length

            if ex_idx < 3 and display_examples:
                print(f"========= Train Example {ex_idx+1} =========")
                print("tokens: %s" % " ".join([str(x) for x in tokens]))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                print("mask: %s" % " ".join([str(x) for x in mask]))
                if not predict:
                    print("label: %s" % label)
                print("")

            if not predict:
                feat = {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_masks': mask, 'label': label}
                if split == "train":
                    if feat["label"] == 1:
                        features_pos.append(feat)
                    else:
                        features_neg.append(feat)
                else:
                    per_claim_features.append(feat)
            else:  # predict
                feat = {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_masks': mask}
                per_claim_features.append(feat)

        if split != "train":
            val_features.append((per_claim_features, example["id"], example["more_than_two"]))

    if split == "train":
        return features_pos, features_neg
    else:
        return val_features


def build_ss_model(args, num_labels=2):
    if args.model == "koelectra":
        return ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator",
                                                                cache_dir=args.cache_dir, num_labels=num_labels)
    else:
        return BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',
                                                             cache_dir=args.cache_dir, num_labels=num_labels)


def main_worker(gpu, train_dataset_pos, train_dataset_neg, val_features, args):
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:9855", rank=args.gpu, world_size=args.n_gpu)

    model = build_ss_model(args, num_labels=2)
    model = model.to(args.gpu)
    model = DDP(model, device_ids=[args.gpu])

    val_idx_dataset = TensorDataset(torch.LongTensor(range(len(val_features))))  # number of docs in validation sets
    val_idx_sampler = DistributedSampler(val_idx_dataset, shuffle=False)
    val_idx_loader = DataLoader(
        val_idx_dataset,
        sampler=val_idx_sampler,
        batch_size=1,
    )
    if args.evaluate:
        dist.barrier()
        validate(val_idx_loader, val_features, model, args)
        return None

    # Training
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.num_train_optimization_steps * args.warmup_proportion),
        num_training_steps=args.num_train_optimization_steps
    )

    if args.multiproc_dist:
        train_sampler_pos = DistributedSampler(train_dataset_pos, shuffle=True)
        train_sampler_neg = DistributedSampler(train_dataset_neg, shuffle=True)
    else:
        train_sampler_pos = RandomSampler(train_dataset_neg)
        train_sampler_neg = RandomSampler(train_dataset_neg)

    train_dataloader_pos = DataLoader(
        train_dataset_pos,
        sampler=train_sampler_pos,
        batch_size=args.batchsize,
    )
    train_dataloader_neg = DataLoader(
        train_dataset_neg,
        sampler=train_sampler_neg,
        batch_size=args.negative_batchsize,
    )

    if args.gpu == 0:
        print("=== Num pos examples : %d" % len(train_dataloader_pos))
        print("=== Batch size : %d" % args.batchsize)
        print("=== Negative Batch size : %d" % args.negative_batchsize)

    for epoch in range(args.epoch_from, args.num_train_epochs + 1):
        args.epoch = epoch
        if args.gpu == 0:
            logger.info(f"\n===== Epoch {epoch} =====")
        if args.multiproc_dist:
            train_sampler_pos.set_epoch(epoch)
            train_sampler_neg.set_epoch(epoch)

        train(train_dataloader_pos, train_dataloader_neg, model, optimizer, scheduler, args)
        dist.barrier()

        # validate(val_idx_loader, val_features, model, args)


def train(train_dataloader_pos, train_dataloader_neg, model, optimizer, scheduler, args):
    # Load temp checkpoint of previous epoch
    if not args.debug:
        temp_ckpt_path = os.path.join(args.checkpoints_dir, f"temp_ckpt.pth")
        if os.path.isfile(temp_ckpt_path):
            temp_ckpt = torch.load(temp_ckpt_path, map_location=f"cuda:{args.gpu}")
            if args.gpu == 0:
                print(f"=== Load checkpoint from (Chunk {temp_ckpt['chunk_num']}, Epoch {temp_ckpt['epoch']})")
            model.module.load_state_dict(temp_ckpt["state_dict"])
            if args.chunk_num == temp_ckpt["chunk_num"]:
                optimizer.load_state_dict(temp_ckpt["optimizer"])
                scheduler.load_state_dict(temp_ckpt["scheduler"])
                current_step = optimizer.state[optimizer.param_groups[0]["params"][-1]]["step"]
                if args.gpu == 0:
                    print("=== Continue optimization from step : %d" % current_step)
                    print("=== total steps : %d" % args.num_train_optimization_steps)
            else:  # args.chunk_num > temp_ckpt["chunk_num"]
                # As new chunk have just started, optimizer should be initialized with default settings
                if args.gpu == 0:
                    print("=== New Chunk! Optimizer will be initialized")
            del temp_ckpt

    model.train()

    if args.gpu == 0:
        tr_loss, num_tr_exs, num_tr_steps = 0, 0, 0
        pbar = tqdm(total=len(train_dataloader_pos), desc="Iteration")

    it = iter(train_dataloader_neg)
    for step, batch in enumerate(train_dataloader_pos):
        batch = tuple(t.to(args.gpu) for t in batch)
        batch_neg = tuple(t.to(args.gpu) for t in next(it))

        input_ids, segment_ids, input_masks, labels = batch
        input_ids_neg, segment_ids_neg, input_masks_neg, labels_neg = batch_neg

        # batchify
        input_ids_cat = torch.cat([input_ids, input_ids_neg], dim=0)
        segment_ids_cat = torch.cat([segment_ids, segment_ids_neg], dim=0)
        input_masks_cat = torch.cat([input_masks, input_masks_neg], dim=0)
        label_ids_cat = torch.cat([labels.view(-1), labels_neg.view(-1)], dim=0)

        model.zero_grad()
        # compute loss and backpropagate
        outputs = model(
            input_ids_cat,
            token_type_ids=segment_ids_cat,
            attention_mask=input_masks_cat,
            labels=label_ids_cat
        )
        loss, logits = outputs.loss, outputs.logits

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if args.gpu == 0:
            tr_loss += loss.item()
            num_tr_exs += input_ids.size(0)
            num_tr_steps += 1

            pbar.update(1)

    dist.barrier()

    # Save checkpoint every epoch
    if args.gpu == 0:
        logger.info("===== Training Done =====")
        print("=== Logging from cuda: %d" % args.gpu)
        print("=== Training loss %f" % (tr_loss / num_tr_steps))

        if not args.debug:
            print(f'=== save checkpoint (Chunk {args.chunk_num}, Epoch {args.epoch})')
            """torch.save({
                'chunk_num': args.chunk_num,
                'epoch': args.epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(args.checkpoints_dir, f"temp_ckpt_chunk{args.chunk_num}_epoch{args.epoch}.pth"))"""
            torch.save({
                'chunk_num': args.chunk_num,
                'epoch': args.epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(args.checkpoints_dir, f"temp_ckpt.pth"))
    dist.barrier()


def check_gpu_memory(gpu):
    GB = 1024 * 1024 * 1024
    t = torch.cuda.get_device_properties(gpu).total_memory
    r = torch.cuda.memory_reserved(gpu)
    a = torch.cuda.memory_allocated(gpu)
    f = r - a  # free inside reserved
    print(f"  {t/GB:.2f} / {r/GB:.2f} / {a/GB:.2f} / {f/GB:.2f}")


def validate(val_idx_loader, val_features, model, args):
    best_ckpt_path = os.path.join(args.checkpoints_dir, "best_ckpt.pth")
    if args.gpu == 0:
        try:
            best_r5 = torch.load(best_ckpt_path, map_location=f"cuda:0")["best_r5"] if os.path.isfile(best_ckpt_path) else 0
        except KeyError:
            best_r5 = 0
        torch.cuda.empty_cache()

    if args.evaluate:
        if args.eval_ckpt:
            eval_ckpt_path = os.path.join(args.checkpoints_dir, args.eval_ckpt)
            eval_ckpt = torch.load(eval_ckpt_path, map_location=f"cuda:{args.gpu}")
        else:
            eval_ckpt = torch.load(best_ckpt_path, map_location=f"cuda:{args.gpu}")
        model.module.load_state_dict(eval_ckpt["model_state"])
    else:
        temp_ckpt_path = os.path.join(args.checkpoints_dir, f"temp_ckpt.pth")
        temp_ckpt = torch.load(temp_ckpt_path, map_location=f"cuda:{args.gpu}")
        model.module.load_state_dict(temp_ckpt["state_dict"])

    check_gpu_memory(args.gpu)

    model.eval()
    val_loss, val_cov5, val_r5 = 0, 0, 0
    if args.gpu == 0:
        pbar = tqdm(total=len(val_idx_loader), desc="Iteration")
        check_gpu_memory(args.gpu)

    for idx in val_idx_loader:
        idx = idx[0].item()
        doc_features, _, more_than_two = val_features[idx]
        doc_losses, doc_logits = [], []

        doc_input_ids = torch.LongTensor([x['input_ids'] for x in doc_features])
        doc_segment_ids = torch.LongTensor([x['segment_ids'] for x in doc_features])
        doc_input_masks = torch.LongTensor([x['input_masks'] for x in doc_features])
        doc_labels = torch.LongTensor([x['label'] for x in doc_features])
        doc_dataset = TensorDataset(doc_input_ids, doc_segment_ids, doc_input_masks, doc_labels)
        doc_dataloader = DataLoader(doc_dataset, batch_size=args.val_batchsize, shuffle=False)

        for batch in doc_dataloader:
            batch = tuple(t.to(args.gpu) for t in batch)
            input_ids, segment_ids, input_masks, labels = batch

            with torch.no_grad():
                outputs = model(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_masks,
                    labels=labels
                )
            loss, logits = outputs.loss, outputs.logits # logits: unnormalized prediction
            doc_losses.append(loss.item())
            doc_logits.append(logits)

        doc_cov5, doc_r5 = calculate_metric(doc_logits, doc_labels.to(args.gpu), idx, more_than_two)

        # doc_cov5: top 5 ss 중 잘 뽑힌 문장 수(여기서 label은 참/거짓이 아님) 대비 전체 
        # doc_r5: 

        val_loss += (sum(doc_losses) / len(doc_losses))
        val_cov5 += doc_cov5
        val_r5 += doc_r5

        if args.gpu == 0:
            pbar.update(1)

    val_loss /= len(val_idx_loader)
    val_cov5 /= len(val_idx_loader)
    val_r5 /= len(val_idx_loader)

    val_results = torch.tensor([val_loss, val_cov5, val_r5]).to(args.gpu)

    dist.barrier()
    dist.reduce(val_results, dst=0, op=dist.ReduceOp.SUM)

    if args.gpu == 0:
        val_results /= args.n_gpu
        val_loss, val_cov5, val_r5 = tuple(val_results.tolist())

        # logging validation results
        split_name = "Test" if args.test else "Validation"
        logger.info(f"===== {split_name} Done =====")
        print(f'=== {split_name} loss', val_loss)
        print(f'=== {split_name} coverage (top5)', val_cov5)
        print(f'=== {split_name} recall (top5)', val_r5)

        if val_r5 > best_r5:
            best_r5 = val_r5
            if not args.debug and not args.evaluate:
                print(f"=== New R@5 record! save best checkpoint "
                      f"(Chunk {temp_ckpt['chunk_num']}, Epoch {temp_ckpt['epoch']})")
                temp_ckpt["best_r5"] = best_r5
                torch.save(temp_ckpt, os.path.join(args.checkpoints_dir, f"best_ckpt.pth"))


def calculate_metric(doc_logits, doc_labels, idx, more_than_two):
    doc_logits_cat = torch.cat(doc_logits, dim=0)
    softmax_logits = F.softmax(doc_logits_cat, 1)[:, 1] # normalization

    true_indices = (doc_labels == 1).nonzero(as_tuple=True)[0] # indeces returned
    n_trues = len(true_indices)
    if n_trues == 0:
        return 0, 0
    # coverage
    five_or_length = min(5, len(softmax_logits))
    _, top5_indices = softmax_logits.topk(five_or_length)
    n_covered_top5 = sum([1 for idx in top5_indices if idx in true_indices])
    coverage_top5 = n_covered_top5 / n_trues # 가장 유사한 5개 문장 중 참인 것 대비 doc label이 1인 애들 수

    # recall
    recall_5 = 1 if ((n_covered_top5 == n_trues and more_than_two == 1) or
                     (n_covered_top5 >= 1 and more_than_two == 0)) else 0

    if five_or_length < 5:
        print(f"# tokens is smaller than 5!!!")
        print(f"\nDocument idx: {idx}, coverage5: {coverage_top5}, "
              f"True_indices: {true_indices}, top_5 indices: {top5_indices}")

    return coverage_top5, recall_5


def main():
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument("--input_dir",
                        default="./data",
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
    parser.add_argument("--temp_dir",
                        default="./ss/tmp/",
                        type=str,
                        help="The temp dir where the processed data file will be saved.")
    parser.add_argument("--checkpoints_dir",
                        default="./ss/checkpoints/",
                        type=str,
                        help="Where checkpoints will be stored.")
    parser.add_argument("--cache_dir",
                        default="./data/models/",
                        type=str,
                        help="Where do you want to store the pre-trained models"
                        "downloaded from pytorch pretrained model.")
    parser.add_argument("--num_chunks",
                        default=10,
                        type=int,
                        help="Dataset chunk size (# of claims) to fit the data into the memory.")
    parser.add_argument("--batchsize",
                        default=1,
                        type=int,
                        help="Batch size for (positive) training examples.")
    parser.add_argument("--negative_batchsize",
                        default=7,
                        type=int,
                        help="Batch size for (negative) training examples.")
    parser.add_argument("--val_batchsize",
                        default=8,
                        type=int,
                        help="Batch size for validation examples.")
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
                        default=3,
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
    parser.add_argument("--eval_ckpt",
                        default="",
                        help="An checkpoint file to evaluate.")
    parser.add_argument('--test',
                        default=False,
                        action='store_true',
                        help='Use test dataset to evaluate.')
    parser.add_argument('--model',
                        default="",
                        type=str,
                        help='Set this as "koelectra" if want to use KoElectra model (https://github.com/monologg/KoELECTRA).')

    args = parser.parse_args()

    if args.multiproc_dist:
        args.n_gpu = torch.cuda.device_count()
        args.gpu = None
    else:
        args.n_gpu = 1
        args.gpu = 0

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
        val_data = load_or_make_data_chunks(args, split="test" if args.test else "val")
        val_features = convert_bert_features(args, val_data, tokenizer, split="val")
        train_dataset_pos, train_dataset_neg = None, None

        if args.multiproc_dist:
            mp.spawn(main_worker, nprocs=args.n_gpu,
                     args=(train_dataset_pos, train_dataset_neg, val_features, args))
        else:
            main_worker(args.gpu, train_dataset_pos, train_dataset_neg, val_features, args)
        return None

    # Train dataset is chunked to avoid out-of-memory issue
    train_data_chunks = load_or_make_data_chunks(args, split="train")
    val_data = load_or_make_data_chunks(args, split="val")

    # training here
    # save the model for each chunk and at load it at the next chunk to learn continuously
    # For training efficiency, run training epochs within each chunk

    args.chunk_from = 1
    args.epoch_from = 1
    if not args.debug:
        temp_ckpt_path = os.path.join(args.checkpoints_dir, f"temp_ckpt.pth")
        if os.path.isfile(temp_ckpt_path):
            temp_ckpt = torch.load(temp_ckpt_path)
            args.chunk_from = temp_ckpt["chunk_num"]
            args.epoch_from = temp_ckpt["epoch"] + 1
            del temp_ckpt
            if args.epoch_from == args.num_train_epochs + 1:
                args.chunk_from += 1
                args.epoch_from = 1

    for chunk_num, train_data in enumerate(train_data_chunks[args.chunk_from-1:], start=args.chunk_from):
        print("\n=====================================")
        print(f"===== Chunk {chunk_num} ... Load the Data =====")
        print("=====================================")
        args.chunk_num = chunk_num
        # number of training steps for each chunk.
        # args.chunk_from: 1일 때, args.num_train_optimization_steps: 40470 
        args.num_train_optimization_steps = (int(
            len([lbl for data in train_data for lbl in data["labels"] if lbl == 1]) / args.batchsize
        ) + 1) * args.num_train_epochs
        # 한 클레임의 candidates 중 레이블이 1인 애들 갯수(cf. len(candidates)==len(laels))
        
        if args.multiproc_dist: #defautl is false
            args.num_train_optimization_steps /= args.n_gpu
        # TensorDataset, sampler
        train_features_pos, train_features_neg = convert_bert_features(args, train_data, tokenizer, split='train')


        all_input_ids_pos = torch.LongTensor([x['input_ids'] for x in train_features_pos])
        all_segment_ids_pos = torch.LongTensor([x['segment_ids'] for x in train_features_pos])
        all_input_masks_pos = torch.LongTensor([x['input_masks'] for x in train_features_pos])
        all_label_pos = torch.LongTensor([x['label'] for x in train_features_pos])
        train_dataset_pos = TensorDataset(all_input_ids_pos, all_segment_ids_pos, all_input_masks_pos, all_label_pos)

        all_input_ids_neg = torch.LongTensor([x['input_ids'] for x in train_features_neg])
        all_segment_ids_neg = torch.LongTensor([x['segment_ids'] for x in train_features_neg])
        all_input_masks_neg = torch.LongTensor([x['input_masks'] for x in train_features_neg])
        all_label_neg = torch.LongTensor([x['label'] for x in train_features_neg])
        train_dataset_neg = TensorDataset(all_input_ids_neg, all_segment_ids_neg, all_input_masks_neg, all_label_neg)

        # run validation every training epoch
        val_features = convert_bert_features(args, val_data, tokenizer, split="val")
        print("===== Data is prepared =====")

        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        if args.multiproc_dist:
            mp.spawn(main_worker, nprocs=args.n_gpu,
                     args=(train_dataset_pos, train_dataset_neg, val_features, args))
        else:
            main_worker(args.gpu, train_dataset_pos, train_dataset_neg, val_features, args)

        args.epoch_from = 1


if __name__ == "__main__":
    print(f"Job is running on {socket.gethostname()}")
    main()