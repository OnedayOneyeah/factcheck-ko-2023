import os
import json
import time
import pickle
import socket
import logging
import datetime
import argparse
from tqdm import tqdm
from collections import defaultdict
from bs4 import BeautifulSoup
import wikipedia
from wikipedia.wikipedia import _wiki_request, search
from urllib3.exceptions import ProtocolError, NewConnectionError
from requests.exceptions import ConnectionError
from multiprocessing.pool import ThreadPool
from functools import partial

wikipedia.set_lang("ko")
logger = logging.getLogger(__name__)


"""
def is_not_multicol(css_class):
    if css_class:
        return "multicol" not in css_class
    else:
        return True
"""

EXCLUDE_LIST = [
    {"name": "style"},
    {"name": "div", "class_": ["dablink hatnote", "thumb tright", "thumb tleft", "mw-references-wrap"]},
    {"name": "div", "id": "toc"},
    {"name": "table"},
    # {"name": "table", "class_": is_not_multicol},
    # {"name": "table", "class_": (lambda x: ("multicol" not in x) or ("wikitable" not in x))},
    {"name": "div", "id": "catlinks"},
    {"name": "div", "role": "navigation"},  # class=["navbox", "navbox authority-control"]
    {"name": "math"},
    {"name": ["ol", "ul", "dl"]},
    {"name": ["h1", "h2", "h3", "h4", "h5", "h6"]}, 
]

# INCLUDE_LIST = ['p', 'span', 'blockquote']


def get_input_files(input_dir, dr_dir):
    with open(os.path.join(input_dir, "wiki_claims.json"), "r") as fp:
        claims = json.load(fp)

    with open(os.path.join(dr_dir, "dr_results.json"), "r") as fp:
        dr_results = json.load(fp)

    return claims, dr_results


def get_output_files(input_dir, corpus_dir):
    if os.path.exists(os.path.join(corpus_dir, f"wiki_docs.json")):
        with open(os.path.join(corpus_dir, f"wiki_docs.json"), "r") as fp:
            corpus = defaultdict(dict, json.load(fp))
    else:
        if not os.path.exists(corpus_dir):
            os.mkdir(corpus_dir)
        corpus = defaultdict(dict)

    if os.path.exists(os.path.join(input_dir, f"warnings.json")):
        with open(os.path.join(input_dir, f"warnings.json"), "r") as fp:
            warnings = defaultdict(dict, json.load(fp))
    else:
        warnings = defaultdict(dict)

    if os.path.exists(os.path.join(input_dir, f"done_list.pickle")):
        with open(os.path.join(input_dir, f"done_list.pickle"), "rb") as fp:
            done_list = pickle.load(fp)
    else:
        done_list = []

    return corpus, warnings, done_list


def make_request(title, date, titles_annotated, corpus):  # returns two list of tuples: new_text and warnings
    warnings = []

    title_anno = title
    j = 1
    while j < 12:
        try:
            # make sure the title is available and correct
            response = _wiki_request({"titles": title, "redirects": ""})
            page_id = list(response['query']['pages'].keys())[0]
            if page_id != '-1':
                norm_check = response['query'].get('normalized')
                if norm_check:
                    norm_title = norm_check[0]['to']
                    title = norm_title
                rd_check = response['query'].get('redirects')
                if rd_check:
                    rd_title = rd_check[0]['to']
                    title = rd_title  # If redirected, change the title
            else:  # If title is wrong, do search
                results, suggestion = search(title, results=1, suggestion=True)
                try:
                    title = suggestion or results[0]
                    page_id = list(_wiki_request({"titles": title})['query']['pages'].keys())[0]
                    assert page_id != -1
                except IndexError or AssertionError:
                    # if there is no suggestion or search results, the page doesn't exist
                    # logger.warning(f"Wrong title: {title} / claim id: {data['id']}")
                    if title_anno in titles_annotated:
                        warnings.append(("wrong_title", {"title": title_anno, "title_searched": title,
                                                         "message": "No search result"}))
                    break

            if title_anno != title and title_anno in titles_annotated:
                warnings.append(("wrong_title", {"title": title_anno, "title_searched": title,
                                                 "message": "title searched"}))

            # find oldid
            date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f") \
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            oldid_params = {
                "action": "query",
                "prop": "revisions",
                "rvprop": "ids|timestamp",
                "rvlimit": 10,
                "rvstart": date_formatted,
                "titles": title
            }
            response2 = _wiki_request(oldid_params)
            rev_check = response2['query']['pages'][page_id].get('revisions')
            if not rev_check:
                if title_anno in titles_annotated:
                    warnings.append(("no_revision_found",
                                     {"title": title_anno, "title_searched": title, "date": date,
                                      "message": "Can't find any revisions of the title"}))
                break
            else:
                break_outer = False
                k = 0
                while True:
                    revid = rev_check[k]['revid']
                    revdate = rev_check[k]["timestamp"]
                    if revdate in corpus[title]:
                        break_outer = True
                        break
                    # Get the content (html format)
                    params = {"action": "parse", "prop": "text", "oldid": revid, "format": "json"}
                    html_check = _wiki_request(params).get('parse')
                    if html_check:
                        html_text = html_check['text']['*']
                        break
                    else:
                        if k < 9:
                            k = k + 1
                            continue
                        else:
                            break_outer = True
                            # logger.warning(f"All the last 10 revisions from date {date} aren't available."
                            #                f"title annotated: {title_anno}, searched: {title}")
                            if title_anno in titles_annotated:
                                warnings.append(("no_revision_found",
                                                 {"title": title_anno, "title_searched": title, "date": date,
                                                  "message": "All the last 10 revisions aren't available"}))
                            break
                if break_outer:
                    break

            soup = BeautifulSoup(html_text, 'lxml')
            for exc in EXCLUDE_LIST:
                for s in soup.find_all(**exc):
                    s.extract()
            valid_text_lst = soup.select("p")
            plain_text = ""
            for val_txt in valid_text_lst:
                plain_text = plain_text + val_txt.text + '\n'
            plain_text = plain_text.replace(u'\xa0', u' ')
            return (title, revdate, plain_text), warnings

        except ConnectionError or ProtocolError or NewConnectionError as E:
            # logger.warning("Connection reset error received! Trial #" + str(j))
            logger.info(E)
            time.sleep(60 * j)
            j += 1
            if j == 12:
                # logger.warning(f"Connection repetitively fails. Skip this title: {title}")
                warnings.append(("connection_error",
                                {"title": title_anno, "title_searched": title,
                                 "message": "Connection repetitively failed."}))
                return None, warnings
    return None, warnings


def download(data_tup, claims, dr_results, done_list, corpus, parallel, n_cpu, production):  # (cid, new_text_list, warning_list)
    cid, data = data_tup

    if cid in done_list:
        return cid, [], []

    new_text_list = []  # list of ("title", "date", "plain_text")
    warning_list = []  # list of ("warning type", {warning info})

    orig_id = cid
    orig_data = data
    while True:
        if orig_data:
            if orig_data["is_variation"] == 0:
                date = orig_data["Date"]
                break
            else:
                orig_id = str(orig_data["original_claim_id"])
                orig_data = claims.get(orig_id)
        else:
            warning_list.append(("wrong_orig_ids",
                                 {"original_claim_id": orig_id, "message": "wrong original claim id"}))
            date = data["Date"]
            break

    titles_annotated = list(set([data[f"title{i}"] for i in range(1, 6) if data[f"title{i}"]]))
    titles_to_download = list(set(dr_results[cid] + titles_annotated))

    multiproc_partial_make_request = partial(make_request, date=date, titles_annotated=titles_annotated, corpus=corpus)
    with ThreadPool(processes=n_cpu if parallel and production else 1) as p:
        for new_text, warnings in (p.imap if parallel and production else map)(
                multiproc_partial_make_request, titles_to_download
        ):
            warning_list.extend(warnings)
            if new_text:
                new_text_list.append(new_text)


    return cid, new_text_list, warning_list



def main(input_dir, dr_dir, corpus_dir, parallel, n_cpu, production, subset_size=100):
    """
    claims: human annotated claims {id: {evidence: ~, ...}, ...}
    dr_results: DR results with predicted titles for each claim {claim_id: [title1, title2, ...], ...}
    done_list: list of ids whose titles are already downloaded
    corpus: {title: {date 1: text 1, date2: text2, ...}, ...}
    warnings: {claim id: {warning_type: {warning_info}}, ...}
    """

    claims, dr_results = get_input_files(input_dir, dr_dir)

    logger.info(f"claims length: {len(claims)}, multiprocessing: {parallel}, subset_size: {subset_size}")
    claims_items = list(claims.items())  # list of [cid, data]

    _, _, done_list = get_output_files(input_dir, corpus_dir)
    done_list = set(done_list)
    claims_items = [item for item in claims_items if item[0] not in done_list]

    pbar = tqdm(range(len(claims_items) // subset_size + 1))
    for i in pbar:
        corpus, warnings, done_list = get_output_files(input_dir, corpus_dir)
        multiproc_partial_download = partial(download, claims=claims, dr_results=dr_results, done_list=done_list,
                                             corpus=corpus, parallel=parallel, n_cpu=n_cpu, production=production)
        pbar.set_description(f"# downloaded pages : {len(corpus)}/???")

        subset_items = claims_items[i * subset_size: (i + 1) * subset_size]
        with ThreadPool(processes=n_cpu if parallel and not production else 1) as p:
            for output in (p.imap if parallel and not production else map)(multiproc_partial_download, subset_items):
                cid, new_text_list, warning_list = output
                for title, rev_date, text in new_text_list:
                    if rev_date not in corpus[title]:
                        corpus[title][rev_date] = text
                for warning_type, warning_info in warning_list:
                    warnings[cid][warning_type] = warning_info
                if cid not in done_list:
                    done_list.append(cid)

        with open(os.path.join(corpus_dir, f"wiki_docs.json"), "w") as fp:
            json.dump(corpus, fp, indent=4, ensure_ascii=False)

        with open(os.path.join(input_dir, f"warnings.json"), "w") as fp:
            json.dump(warnings, fp, indent=4, ensure_ascii=False)

        with open(os.path.join(input_dir, f"done_list.pickle"), "wb") as fp:
            pickle.dump(done_list, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data")
    parser.add_argument("--dr_dir", type=str, default="./dr")
    parser.add_argument("--corpus_dir", type=str, default="./data/wiki")
    parser.add_argument("--n_cpu", type=int, default=None)
    parser.add_argument("--parallel", default=False, action="store_true")
    parser.add_argument("--production", default=False, action="store_true")
    parser.add_argument("--subset_size", type=int, default=10)
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter('%(asctime)s - {} - %(levelname)s - %(message)s'.format(socket.gethostname())))
    logger.addHandler(stream_handler)

    main(**vars(args))
