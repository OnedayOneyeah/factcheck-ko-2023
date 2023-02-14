import os
import json
import argparse


def main(input_dir, dr_dir, print_samples):
    with open(os.path.join(input_dir, "wiki_claims.json"), "r") as fp:
        dataset = json.load(fp)

    with open(os.path.join(dr_dir, "dr_results.json"), "r") as fp:
        dr_results = json.load(fp)

    scores = []
    TITLES = ['title1', 'title2', 'title3', 'title4', 'title5']
    no_title_count = 0  # number of True/False claims label that has no titles
    for i, _id in enumerate(dataset):
        titles_annotated = set([dataset[_id][title] for title in TITLES if dataset[_id][title] is not None])
        count = 0  # number of correctly retrieved titles

        missed_titles = []
        for title in titles_annotated:
            if title.strip() in dr_results[_id]:
                count += 1
            else:
                if print_samples and i <= 5:
                    missed_titles.append(title)

        if print_samples and i <= 5:
            print(f"Claim: {dataset[_id]['claim']}")
            print(f"Titles Annotated: {list(titles_annotated)}")
            print(f"DR results: {dr_results[_id]}")
            print(f"Missed Titles {missed_titles}")
            print()

        try:
            scores.append(1 if ((count == len(titles_annotated) and dataset[_id]["more_than_two"] == 1)
                                or (count >= 1 and dataset[_id]["more_than_two"] == 0)) else 0)
        except ZeroDivisionError:  # For NEI claims, there's no titles
            if dataset[_id]['True_False'] != 'None':
                no_title_count += 1

    total_n_titles = 0
    for cid in dr_results:
        total_n_titles += len(dr_results[cid])

    print("\nDR score:", sum(scores)/len(scores))
    print("Avg # titles:", total_n_titles/len(dr_results))
    print("# of no titiles annotated:", no_title_count, "/", len(dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data", help="input directory that contains 'wiki_claims.json'")
    parser.add_argument("--dr_dir", type=str, default="./dr", help="directory where result of DR will be saved.")
    parser.add_argument("--print_samples", default=False, action="store_true")
    args = parser.parse_args()

    main(**vars(args))
