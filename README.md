# factcheck-ko-2021

Based on the project in 2020: https://github.com/ozmig77/factcheck-ko-2020


# Install
```
git clone https://github.com/hongcheki/factcheck-ko-2021.git
cd factcheck-ko-2021
conda create -n factcheck-ko-2021 python=3.8
conda activate factcheck-ko-2021
pip install -r requirements.txt
```


# Demo
1. Download the pretrained checkpoints.
- Save SS checkpoint [(download)](https://drive.google.com/file/d/1-XuWTl2PKtfrCJMwxwlhq91O9xr86VVp/view?usp=sharing) in `ss/checkpoints`
- Save RTE checkpint [(download)](https://drive.google.com/file/d/14InhVylKC05i2POo6gGBNXjb9EDp2Nlk/view?usp=sharing) in `rte/checkpoints`.

2. Run demo.py, i.e., `python demo.py` Make sure your gpu is available.

3. Test the model with your own claim.


# Training

### Data
Download the data for training [here](https://drive.google.com/drive/folders/1cYJejZ6gxT7TARy7BtWN77384VgYmjoE?usp=sharing). Save the files in `./data/`
- `data/wiki_claims.json`: Human-Annotated Dataset for the Factcheck
- `data/train_val_test_ids.json`: Lists of claim ids for train/validation/test split

### Document Retrieval

1. Document retrieval
    ```
    python dr/document_retrieval.py --parallel
    ```
    This will create `dr/dr_results.json`. **Warning!! It takes a long time.**\
    _(Our `dr_results.json` is [here](https://drive.google.com/file/d/1QWrYORC3udpgOQ5ZmXzg7fAcHcR3kM7H/view?usp=sharing))_

2. Evaluate DR
    ```
    python dr/dr_scoring.py --print_samples
    ```
    This will print the DR scores and some information to the prompt.\
    Our result: 84.18% (recall, entire dataset).

### Sentence Selection

1. Download Wikipedia documents\
    We should download the Wikipedia documents whose titles are retrieved in DR.
    ```
    python wiki_downloader.py --parallel
    ```
    This will create
    - `data/wiki/wiki_docs.json`: Wikipedia documents corresponing to claims in `wiki_claims.json`
    - `data/warnings.json` : Set of data that lacks the integrity. (e.g. Typo in annotated title)
    - `data/done_list.json` : List of claim ids whose documents are already downloaded. Handling interruptions or errors.

    \
    **Warning!! It takes more than 70 hours with 50000 claims.**\
    _(Our `wiki_docs.json` is [here](https://drive.google.com/file/d/1q4cYyLEPGF84-F3ihci9rqXG7eJU9SlL/view?usp=sharing))_

2. Training SS model
    ```
    python ss/train_ss.py
    ```
    This will create `ss/checkpoints/best_ckpt.pth`.

3. Evaluate SS model with test data
    ```
    python ss/train_ss.py --evaluate --test
    ```
    Our result: 49.29% (recall with top5 sentence, for test data).

### Recognizing Textual Entailment

1. Get NEI SS results\
    NEI claims don't have gold evidences, thus we need to feed the RTE model with results of Sentence Selection for NEI claims.
    ```
    python ss/get_nei_ss_results.py
    ```
    This will create `./ss/nei_ss_results.json`.\
    _(Our `nei_ss_results.json` is [here](https://drive.google.com/file/d/1t9MkhoqNhRCStBKSIHG1a4rbLYg7f4kG/view?usp=sharing))_

2. Training RTE model
    ```
    python rte/train_rte.py
    ```
    This will create `rte/checkpoints/best_ckpt.pth`.

3. Evaluate RTE model
    ```
    python rte/train_rte.py --evaluate --test
    ```
    Our result: 64.67% (accuracy, for test data).
