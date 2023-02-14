import argparse
import os
import json
import time
from tqdm import tqdm
import wikipedia
wikipedia.set_lang("ko")
from konlpy import tag
from multiprocessing.pool import ThreadPool

from requests.exceptions import ConnectionError
from urllib3.exceptions import ProtocolError


class DocRetrieval:
    def __init__(self, k_wiki_results, tagger, parallel, n_cpu, production):
        self.k_wiki_results = k_wiki_results
        self.tagger = getattr(tag, tagger)()
        self.parallel = parallel
        self.n_cpu = n_cpu
        self.production = production

    '''def get_nouns(self, line):
        # Get all nouns of each claim
        claim = line["claim"]
        nouns = self.tagger.nouns(claim)
        return nouns'''

    def get_uni_and_bigrams(self, claim):
        claim = claim.strip() + " "
        claim_pos = self.tagger.pos(claim)
        new_claim = ""
        position = 0
        buffer = ""
        store = False
        for i in range(len(claim_pos)):
            word, pos = claim_pos[i]

            if pos not in ["Josa", "Punctuation", "Foreign"]:
                buffer += word

            if pos in ["Noun", "Number", "Alpha"]:
                store = True

            position += len(word)
            if claim[position] == " ":
                new_claim = new_claim + buffer + " " if store else new_claim
                buffer = ""
                store = False
                position += 1

        unigrams = new_claim.split()
        bigrams = [" ".join(unigrams[i:i+2]) for i in range(len(unigrams) - 1)]
        return unigrams + bigrams

    def search(self, s_input):
        if len(s_input) > 100:
            print(f"search input is extraordinarily long. Please check! \ninput: {s_input}")
            return []
        i = 1
        while i < 12:
            try:
                docs, suggestion = wikipedia.search(s_input, suggestion=True)
                if suggestion:
                    docs.append(suggestion)
                if self.k_wiki_results is not None:
                    return docs[:self.k_wiki_results]
                else:
                    return docs
            except (ConnectionError, ProtocolError, wikipedia.exceptions.WikipediaException):
                print("Connection reset error received! Trial #" + str(i))
                time.sleep(60 * i)
                i += 1
        return []

    def get_doc_for_claim(self, claim):
        # nouns = self.get_nouns(line)
        search_inputs = self.get_uni_and_bigrams(claim) + [claim[:100]]
        predicted_pages = []  # list of wiki document titles

        with ThreadPool(processes=self.n_cpu if self.parallel and self.production else None) as p:
            for s_result in (p.imap_unordered if self.parallel and self.production else map)(
                    lambda s_input: self.search(s_input), search_inputs
            ):
                predicted_pages.extend(s_result)
        predicted_pages = list(set(predicted_pages))
        return search_inputs, predicted_pages

    def get_output(self, data_tup, done_list):
        data_id, data = data_tup
        if data_id in done_list:
            return None
        else:
            _, predicted_pages = self.get_doc_for_claim(data["claim"])
            return data_id, predicted_pages


def main(input_dir, dr_dir, k_wiki, tagger, parallel=False, n_cpu=None, production=False):
    in_file = os.path.join(input_dir, "wiki_claims.json")
    out_file = os.path.join(dr_dir, "dr_results.json")

    dr_model = DocRetrieval(k_wiki, tagger, parallel, n_cpu, production)
    with open(os.path.join(in_file), "r") as fp:
        dataset = json.load(fp)

    if os.path.isfile(os.path.join(out_file)):
        with open(os.path.join(out_file), "r") as fp:
            dr_results = json.load(fp)
    else:
        dr_results = {}
    done_list = dr_results.keys()

    dataset_items = list(dataset.items())

    with ThreadPool(processes=n_cpu if parallel and not production else 1) as p:
        for outputs in tqdm(
            (p.imap_unordered if parallel and not production else map)(
                lambda data_tup: dr_model.get_output(data_tup, done_list), dataset_items
            ),
            total=len(dataset)
        ):
            if outputs:
                data_id, predicted_pages = outputs
                dr_results[data_id] = predicted_pages

    with open(os.path.join(out_file), "w") as fp:
        json.dump(dr_results, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data", help="input directory that contains 'wiki_claims.json'")
    parser.add_argument("--dr_dir", type=str, default="./dr", help="directory where result of DR will be saved.")
    parser.add_argument("--k_wiki", type=int, default=1, help="first k pages for wiki search")
    parser.add_argument("--tagger", type=str, default="Okt", help="KoNLPy tagger. Strongly recommend to use the default.")
    parser.add_argument("--parallel", default=False, action="store_true")
    parser.add_argument("--n_cpu", type=int, default=None, help="number of cpus for multiprocessing")
    parser.add_argument("--production", default=False, action="store_true", help="If true, multiprocessing is done in each claim")
    args = parser.parse_args()

    print(vars(args))
    if not os.path.exists(args.dr_dir):
        os.mkdir(args.dr_dir)

    main(**vars(args))