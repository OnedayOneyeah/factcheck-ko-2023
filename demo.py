from eval import *

import json
import random
import openpyxl
import warnings
warnings.filterwarnings("ignore")

label_dict = {0: 'True', 1: 'False', 2: 'NEI'}

def get_wiki_titles(args):
    excel_doc = openpyxl.load_workbook(args.wiki_titles_excel_fname)
    sheet = excel_doc.get_sheet_by_name("Sheet1")
    wiki_titles = [row[0].value for row in sheet.rows if row[0].value]
    
    return wiki_titles

def get_news_titles(args):
    news_titles = json.load(open(args.news_titles_json_fname, 'r', encoding="utf-8"))

    return news_titles

if __name__ == "__main__":
    print(f"Job is running on {socket.gethostname()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed",
                        type=int,
                        default=1234)
    parser.add_argument("--wiki_titles_excel_fname",
                        type=str,
                        default="data/for_demo/factcheck_wiki_titles.xlsx")
    parser.add_argument("--news_titles_json_fname",
                        type=str,
                        default="data/for_demo/factcheck_news_titles.json")
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
                        default="./ss/checkpoints/",
                        type=str,
                        help="Where checkpoints for SS will be / is stored.")
    parser.add_argument("--ss_checkpoint",
                        default="best_ckpt.pth",
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
                        default="./rte/checkpoints/",
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

    random.seed(args.random_seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.parallel = ~args.non_parallel
    args.n_cpu = args.n_cpu if args.parallel else 1
    #if args.rte_model == "koelectra":
    #    args.rte_checkpoints_dir = os.path.dirname(args.rte_checkpoints_dir) + "_" + args.rte_model

    # for dictionary; wiki_titls = List[Subject]
    # e.g. ['포도주', '인공지능', ...]
    wiki_titles = get_wiki_titles(args)
    
    # for news;
    # {
    #   'Subjects': {
    #       '라디오': ["AKR20210101008600063", ...],
    #       ...
    #   },
    #   'ids': {
    #       "AKR20210101008600063": [
    #           'TBS 라디오, 개국 31주년 특집 방송',            # title
    #           'http://yna.kr/AKR20210610160100005',          # url
    #           'TBS는 라디오 개국 31주년을 맞아 ...'           # text
    #       ],
    #        ...
    #   },
    # }
    news_titles = get_news_titles(args)

    # Build pipelines of each step
    print("Load the models... This takes about 40 seconds...")
    dr_pipeline = DocumentRetrieval(args)
    ss_pipeline = SentenceSelection(args)  # Loading the BERT model for SS
    rte_pipeline = RecognizeTextualEntailment(args)  # Loading the BERT/ELECTRA model for RTE
    while True:
        # choose claim type: (d)ictionary or (n)ews
        claim_type = input("Claim from (d)ictionary or (n)ews: ")
        while claim_type not in ['d', 'n']:
            claim_type = input("Claim from (d)ictionary or (n)ews: ")

        if claim_type == 'd':
            # randomly select 20 titles
            random.shuffle(wiki_titles)
            ex_titles = wiki_titles[:20]
            print(ex_titles)
        else:
            # randomly select 30 subjects
            news_subjects = list(news_titles['Subjects'].keys())
            random.shuffle(news_subjects)
            ex_subjects = news_subjects[:30]

            # for each subject, choose one news
            ex_news = {
                ex_subject: news_titles['ids'][random.choice(news_titles['Subjects'][ex_subject])]
                for ex_subject in ex_subjects
            }
            
            for k,v in ex_news.items():
                print(k)
                print(v)
                print()

            selected_subject = input("Choose one subject: ")
            while selected_subject not in ex_news:
                selected_subject = input("Choose one subject: ")
            
            selected_title, selected_url, selected_text = ex_news[selected_subject]
            print(selected_title)
            print(selected_url)
            print(selected_text)

        try:
            claim = input("Please put in a claim you want to check: ")
        except Exception as E:
            print(E)
            continue

        if not claim:
            continue

        start = time.time()
        # DR
        if claim_type == 'd':
            dr_results = dr_pipeline.get_dr_results(claim)
        else:
            dr_results = {selected_title: selected_text}
        dr_end = time.time()
        print("\n========== DR ==========")
        print(f"DR results: {', '.join(dr_results)}")
        print(f"DR Time taken: {dr_end - start:0.2f} (sec)")

        # SS
        ss_scores, ss_titles, ss_results = ss_pipeline.get_results(claim, dr_results)
        ss_results_print = "\n".join([
            f"{ss_title}: {ss_result} ({ss_score})"
            for ss_title, ss_result, ss_score
            in zip(ss_titles, ss_results, ss_scores)])
        ss_end = time.time()
        print("\n========== SS ==========")
        print(f'SS results: \n{ss_results_print}')
        print(f"SS Time taken: {ss_end - dr_end:0.2f} (sec)")

        # RTE
        predicted_label, rte_score = rte_pipeline.get_results(claim, ss_results)
        rte_end = time.time()
        print("\n========== RTE ==========")
        print(f"Predicted Label: {label_dict[predicted_label]} ({rte_score})")
        print(f"RTE Time taken: {rte_end - ss_end:0.2f} (sec)")
        print("\n=========================\n")
