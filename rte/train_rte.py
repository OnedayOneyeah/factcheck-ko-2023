import os
import re
import json
import pickle
import random
import argparse
import logging
import socket
from collections import defaultdict
import numpy as np
from tqdm import tqdm
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


def get_dataset(args, split):
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

    examples = []
    warnings = defaultdict(dict)
    name2label = {'True': 0, 'False': 1, 'None': 2}
    for cid in claims:
        data = claims[cid]
        claim = data['claim']
        label = name2label[data['True_False']]

        if label != 2:
            evidences_anno = [data[f"evidence{i}"] for i in range(1, 6) if data[f"evidence{i}"]]
            # 여기서 evidence는 human-annotated된 근거 자료들을 의미함
            evidences_clean = [ev_clean for ev_anno in evidences_anno
                               for ev_clean in cleanse_and_split(ev_anno) if len(ev_clean) >= 10]
            if len(evidences_clean) == 0:
                warnings[cid]["No evidences annotated but not NEI"] = [claim]
                continue

        else:  # NEI -> use ss results (ss/dir_name/nei_ss_results.json)
            evidences_clean = nei_ss_results[cid][:2]

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


def convert_dataset(examples, tokenizer, max_length, split, predict=False, display_examples=False):
    """
    Convert train examples into BERT's input foramt.
    """
    features = []
    for ex_idx, example in tqdm(enumerate(examples), total=len(examples)):
        sentence_b = tokenizer.tokenize(example['claim'])
        if split != "train":  # "val" or "test"
            per_claim_features = []
        if not predict:
            label = example['label']

        for idx in range(len(example['evidences'])):
            ev = example['evidences'][idx]
            sentence_a = tokenizer.tokenize(ev)

            if len(sentence_a) + len(sentence_b) > max_length - 3:  # 3 for [CLS], 2x[SEP]
                logger.warning(
                    "The length of the input is longer than max_length! "
                    f"sentence_a: {sentence_a} / sentence_b: {sentence_b}"
                )
                # truncate sentence_b to fit in max_length
                diff = (len(sentence_a) + len(sentence_b)) - (max_length - 3)
                sentence_b = sentence_b[:-diff]

            tokens = ["[CLS]"] + sentence_a + ["[SEP]"] + sentence_b + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
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
                if split == "train":
                    features.append(feat)
                else:
                    per_claim_features.append(feat)
            else:  # predict
                feat = {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': mask}
                per_claim_features.append(feat)

        if split != "train":
            features.append(per_claim_features)

    return features


def load_or_make_features(args, tokenizer, split, save=True):
    small = '_small' if args.debug else ''
    features_path = os.path.join(args.temp_dir, f"{split}_features{small}.pickle")

    try:
        with open(features_path, "rb") as fp:
            features = pickle.load(fp)
        return features

    except FileNotFoundError or EOFError:
        dataset = get_dataset(args, split=split)
        features = convert_dataset(dataset, tokenizer, args.max_length, split=split)
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

    model = build_rte_model(args, num_labels=3)
    model = model.to(args.gpu)
    model = DDP(model, device_ids=[args.gpu])

    val_idx_dataset = TensorDataset(torch.LongTensor(range(len(val_features))))  # number of docs in validation sets
    val_idx_sampler = DistributedSampler(val_idx_dataset, shuffle=True)
    val_idx_loader = DataLoader(
        val_idx_dataset,
        sampler=val_idx_sampler
    )

    if args.evaluate:
        dist.barrier()
        ckpt_path = os.path.join(args.checkpoints_dir, "best_ckpt.pth")
        checkpoint = torch.load(ckpt_path, map_location=f"cuda:{args.gpu}")
        model.module.load_state_dict(checkpoint["state_dict"])
        validate(val_idx_loader, val_features, model, None, None, args)
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
        ckpt_path = os.path.join(args.checkpoints_dir, "best_ckpt.pth")
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

        validate(val_idx_loader, val_features, model, optimizer, scheduler, args)


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


def validate(val_idx_loader, val_features, model, optimizer, scheduler, args):
    model.eval()
    val_loss, val_acc = 0, 0
    n_claims_each_label = torch.LongTensor([0, 0, 0])  # Supported, Refuted, NEI
    n_corrects_each_label = torch.LongTensor([0, 0, 0])
    n_predicted_each_label = torch.LongTensor([0, 0, 0])

    if args.gpu == 0:
        print("=== Running validation ===")
        pbar = tqdm(total=len(val_idx_loader), desc="Iteration")

    for idx in val_idx_loader:
        idx = idx[0].item()
        doc_features = val_features[idx]
        doc_losses, doc_logits = [], []

        doc_input_ids = torch.LongTensor([x['input_ids'] for x in doc_features])
        doc_segment_ids = torch.LongTensor([x['segment_ids'] for x in doc_features])
        doc_input_mask = torch.LongTensor([x['input_mask'] for x in doc_features])
        doc_labels = torch.LongTensor([x['label'] for x in doc_features])
        doc_dataset = TensorDataset(doc_input_ids, doc_segment_ids, doc_input_mask, doc_labels)
        doc_dataloader = DataLoader(doc_dataset, batch_size=len(doc_features), shuffle=False)

        for batch in doc_dataloader:
            batch = tuple(t.to(args.gpu) for t in batch)
            input_ids, segment_ids, input_mask, labels = batch

            with torch.no_grad():
                outputs = model(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=labels
                )
            loss, logits = outputs.loss, outputs.logits
            doc_losses.append(loss.item())
            doc_logits.append(logits)

        is_correct, pred = calculate_metric(doc_logits, doc_labels.to(args.gpu))

        val_loss += (sum(doc_losses) / len(doc_losses))
        val_acc += is_correct

        n_corrects_each_label[doc_labels[0]] += is_correct
        n_claims_each_label[doc_labels[0]] += 1
        n_predicted_each_label[pred] += 1

        if args.gpu == 0:
            pbar.update(1)

    val_loss /= len(val_idx_loader)
    val_acc /= len(val_idx_loader)

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

        print()
        print("===== Validation loss", val_loss)
        print("===== Validation accuracy", val_acc)

        recall_each_label = n_corrects_each_label / n_claims_each_label
        precision_each_label = n_corrects_each_label / n_predicted_each_label

        print('===== Recall for each Labels =====')
        print(f'label0 (Supported): {recall_each_label[0]} ({n_corrects_each_label[0]}/{n_claims_each_label[0]})')
        print(f'label1 (Refuted)  : {recall_each_label[1]} ({n_corrects_each_label[1]}/{n_claims_each_label[1]})')
        print(f'label2 (NEI)      : {recall_each_label[2]} ({n_corrects_each_label[2]}/{n_claims_each_label[2]})')

        print('===== Precision for each Labels =====')
        print(f'label0 (Supported): {precision_each_label[0]} ({n_corrects_each_label[0]}/{n_predicted_each_label[0]})')
        print(f'label1 (Refuted)  : {precision_each_label[1]} ({n_corrects_each_label[1]}/{n_predicted_each_label[1]})')
        print(f'label2 (NEI)      : {precision_each_label[2]} ({n_corrects_each_label[2]}/{n_predicted_each_label[2]})')

        if not args.evaluate and val_acc > args.best_acc:
            args.best_acc = val_acc
            if not args.debug:
                print(f'===== New Accuracy record! save checkpoint (epoch: {args.epoch + 1})')
                torch.save({
                    'epoch': args.epoch + 1,
                    'best_acc': args.best_acc,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, os.path.join(args.checkpoints_dir, "best_ckpt.pth"))


def calculate_metric(doc_logits, doc_labels, mode="sum"):
    assert len(set(doc_labels.tolist())) == 1
    label = doc_labels[0]

    doc_logits_cat = torch.cat(doc_logits, dim=0)
    softmax_logits = F.softmax(doc_logits_cat, dim=1)  # e.g. [[0.1, 0.4, 0.5], [0.8, 0.1, 0.1]]
    pred_for_each_claim = softmax_logits.argmax(dim=1)  # e.g. [2, 0]  ... [NEI, True] respectively

    if mode == "fever":
        # At least one supporting evidence => Label as Supported (0)
        # At least one rejecting evidence without supporting one => Label as Refuted (1)
        # Otherwise => Label as NEI (2)
        if 0 in pred_for_each_claim:
            pred = 0
        else:
            if 1 in pred_for_each_claim:
                pred = 1
            else:
                pred = 2
    elif mode == "majority_vote":
        # Simple majority vote
        # if draw, priority is Supported (0) > Refuted (1) > NEI (2)
        pred = int(pred_for_each_claim.mode().values)
    elif mode == "sum":
        pred = softmax_logits.sum(dim=0).argmax().item()

    return int(label == pred), pred


def main():
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument("--input_dir",
                        default="./data/",
                        type=str,
                        help="The input data dir.")
    parser.add_argument("--ss_dir",
                        default="./ss/",
                        type=str,
                        help="To get the ss results of NEI data.")
    parser.add_argument("--temp_dir",
                        default="./rte/tmp/",
                        type=str,
                        help="The temp dir where the processed data file will be saved.")
    parser.add_argument("--checkpoints_dir",
                        default="./rte/checkpoints/",
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
        all_label = torch.LongTensor([x['label'] for x in train_features])
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