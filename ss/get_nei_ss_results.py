import os
import json
import pickle
import random
import argparse
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizerFast

# for multi-gpu training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from train_ss import cleanse_and_split, convert_bert_features, build_ss_model

import warnings
warnings.filterwarnings("ignore")
from transformers import logging as trans_logging
trans_logging.set_verbosity_error()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_or_make_nei_data(args, chunksize=2000, save=True):
    data_path = os.path.join(args.temp_dir, "nei_data.pickle")
    try:
        with open(data_path, "rb") as fp:
            data = pickle.load(fp)
    except FileNotFoundError or EOFError:
        data = get_nei_data(args.input_dir, args.dr_dir, args.corpus_dir)
        if save:
            with open(data_path, "wb") as fp:
                pickle.dump(data, fp)

    output_path = os.path.join(args.output_dir, "nei_ss_results.json")
    if os.path.exists(output_path):
        with open(output_path, "r") as fp:
            results_with_text = json.load(fp)
            chunk_from = len(results_with_text) // chunksize
            done_list = results_with_text.keys()

    data = [d for d in data if d['id'] not in done_list]
    nei_data_chunk = [data[i: i + chunksize] for i in range(0, len(data), chunksize)]
    return nei_data_chunk, chunk_from


def get_nei_data(input_dir, dr_dir, corpus_dir):
    print(f"Make NEI data")

    with open(os.path.join(input_dir, "wiki_claims.json"), "r") as fp:
        claims = json.load(fp)

    with open(os.path.join(dr_dir, "dr_results.json"), "r") as fp:
        dr_results = json.load(fp)

    with open(os.path.join(corpus_dir, "wiki_docs.json"), "r") as fp:
        wiki = json.load(fp)
        wiki_titles = wiki.keys()

    claims = {cid: data for cid, data in claims.items() if data["True_False"] == "None"}

    data_list = []
    for cid in claims:
        data = claims[cid]
        predicted_titles = [title for title in dr_results[cid] if title in wiki_titles]
        candidates = []
        for title in predicted_titles:
            documents = wiki[title]
            date = datetime.datetime.strptime(data["Date"], "%Y-%m-%d %H:%M:%S.%f")
            doc_dates = [datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%SZ") for dt in documents.keys()]
            doc_dates = [dt for dt in doc_dates if dt <= date]
            if not doc_dates:
                # warnings[cid]["Wrong Date"] = {"Date annotated": data["Date"],
                #                                "Dates in downloaded wiki documents": [d.strftime("%Y-%m-%d %H:%M:%S.%f")
                #                                                                       for d in doc_dates]}
                continue
            text = documents[max(doc_dates).strftime("%Y-%m-%dT%H:%M:%SZ")]
            text_lst = cleanse_and_split(text)
            candidates.extend(text_lst)

        claim = data['claim']
        data_list.append({
            'id': cid,
            'claim': claim,
            'candidates': candidates,
            'more_than_two': data["more_than_two"]
        })

    total_n_sentences = 0
    for d in data_list:
        total_n_sentences += len(d["candidates"])

    print(f"<<NEI data ss labelling results>>")
    print(f"Total # NEI claims: {len(data_list)}")
    print(f"Average # sentences: {total_n_sentences / len(data_list)}")
    return data_list


def main_worker(gpu, nei_features, nei_data, args):
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:9855", rank=args.gpu, world_size=args.n_gpu)

    model = build_ss_model(args, num_labels=2)
    model = model.to(args.gpu)
    model = DDP(model, device_ids=[args.gpu])

    n_nei_samples = len(nei_features)
    nei_idx_dataset = TensorDataset(torch.LongTensor(range(n_nei_samples)))
    nei_idx_sampler = DistributedSampler(nei_idx_dataset, shuffle=False)
    nei_idx_loader = DataLoader(
        nei_idx_dataset,
        sampler=nei_idx_sampler,
        batch_size=1,
    )

    nei_results = get_nei_result(nei_idx_loader, nei_features, model, args)

    total_results = [None for _ in range(args.n_gpu)]
    dist.all_gather_object(total_results, nei_results)
    total_results = [r for rank_results in total_results for r in rank_results]

    # Save the results
    if args.gpu == 0:
        output_path = os.path.join(args.output_dir, "nei_ss_results.json")
        try:
            with open(output_path, "r") as fp:
                results_with_text = json.load(fp)
        except FileNotFoundError or EOFError:
            results_with_text = dict()

        total_results_dict = {cid: result for cid, result in total_results}
        for d in nei_data:
            results_with_text[d["id"]] = []
            cand_ids = total_results_dict[d["id"]]
            for cand_id in cand_ids:
                results_with_text[d["id"]].append(d["candidates"][cand_id])

        with open(output_path, "w") as fp:
            json.dump(results_with_text, fp)


def get_nei_result(nei_idx_loader, nei_features, model, args):
    checkpoint = torch.load(os.path.join(args.checkpoints_dir, args.checkpoint),
                            map_location=f"cuda:{args.gpu}")
    best_r5 = checkpoint["best_r5"]
    if args.gpu == 0:
        print(f"Use checkpoint: {os.path.join(args.checkpoints_dir, args.checkpoint)}")
        print(f"R@5 score: {best_r5}")

    model.module.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if args.gpu == 0:
        pbar = tqdm(total=len(nei_idx_loader), desc="Iteration")

    nei_results = []
    for idx in nei_idx_loader:
        idx = idx[0].item()
        doc_features, cid, more_than_two = nei_features[idx]
        doc_logits = []

        doc_input_ids = torch.LongTensor([x['input_ids'] for x in doc_features])
        doc_segment_ids = torch.LongTensor([x['segment_ids'] for x in doc_features])
        doc_input_mask = torch.LongTensor([x['input_masks'] for x in doc_features])
        doc_dataset = TensorDataset(doc_input_ids, doc_segment_ids, doc_input_mask)
        doc_dataloader = DataLoader(doc_dataset, batch_size=args.val_batchsize, shuffle=False)

        for batch in doc_dataloader:
            batch = tuple(t.to(args.gpu) for t in batch)
            input_ids, segment_ids, input_mask = batch

            with torch.no_grad():
                outputs = model(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
            doc_logits.append(outputs.logits)

        doc_logits_cat = torch.cat(doc_logits, dim=0)
        softmax_logits = F.softmax(doc_logits_cat, 1)[:, 1]
        _, top5_indices = softmax_logits.topk(min(5, len(softmax_logits)))
        nei_results.append((cid, top5_indices.tolist()))

        if args.gpu == 0:
            pbar.update(1)

    return nei_results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        default="./data/",
                        type=str,
                        help="The input data dir.")
    parser.add_argument("--dr_dir",
                        default="./dr/",
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
    parser.add_argument("--checkpoint",
                        default="best_ckpt.pth",
                        type=str,
                        help="Which checkpoint to use.")
    parser.add_argument("--output_dir",
                        default="./ss/",
                        type=str,
                        help="The output dir where the model predictions will be stored.")
    parser.add_argument("--cache_dir",
                        default="./data/models/",
                        type=str,
                        help="Where do you want to store the pre-trained models"
                        "downloaded from pytorch pretrained model.")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="random seed.")
    parser.add_argument("--max_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after tokenized."
                        "If longer than this, it will be truncated, else will be padded.")
    parser.add_argument("--val_batchsize",
                        default=8,
                        type=int,
                        help="Batch size for validation examples.")
    parser.add_argument('--multiproc_dist',
                        default=False,
                        action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs.')
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    nei_data_chunks, chunk_from = load_or_make_nei_data(args, chunksize=2000, save=True)

    for chunk_num, nei_data in enumerate(nei_data_chunks, start=chunk_from+1):
        print("\n=====================================")
        print(f"===== Chunk {chunk_num} ... Load the Data =====")
        print("=====================================")

        nei_features = convert_bert_features(args, nei_data, tokenizer, split="val", predict=True)

        if args.multiproc_dist:
            mp.spawn(main_worker, nprocs=args.n_gpu, args=(nei_features, nei_data, args))
        else:
            main_worker(args.gpu, nei_features, nei_data, args)


if __name__ == "__main__":
    main()