# factcheck-ko-2023
- Paper is available [here]https://www.notion.so/Fact-check-automation-b40c30af59a9412caff8da59e4e12921

- Based on the project in 2020: https://github.com/ozmig77/factcheck-ko-2020
- Based on the project in 2021: https://github.com/hongcheki/factcheck-ko-2021

## Leaderboard

|Rank|Recall(%)|SS|RTE|
|---|---|---|---|
|1|60.36|org|mpe(noised)-cls|
|2(Baseline)|58.72|org|spe-cls|
|3|56.16|org|mpe(noised)-rgs|
|4|54.06|org|mpe-rgs|
|5|52.58|org|spe-rgs|
|6|50.88|org|mpe-cls|
|7|48.82|KNN(k=5)|spe-cls|
|8|43.54|KNN(k=5)|mpe(noised)-cls|
|9|40.01|KNN(k=5)|mpe(noised)-rgs|
|10|37.72|KNN(k=5)|mpe-cls|

# Install
```
git clone https://github.com/hongcheki/factcheck-ko-2021.git
cd factcheck-ko-2021
conda create -n factcheck-ko-2021 python=3.8
conda activate factcheck-ko-2021
pip install -r requirements.txt
```

# Data

Download the data for training [here](https://drive.google.com/drive/folders/1cYJejZ6gxT7TARy7BtWN77384VgYmjoE?usp=sharing). Save the files in `./data/`
- `data/wiki_claims.json`: Human-Annotated Dataset for the Factcheck
- `data/train_val_test_ids.json`: Lists of claim ids for train/validation/test split

# Train/Evaluate SS/RTE model

- Documents are available [here](https://github.com/hongcheki/factcheck-ko-2021)

# Evaluate fact-check model

## Demo
1. Download the pretrained checkpoints.
- Save SS checkpoint [(download)](https://drive.google.com/file/d/1-XuWTl2PKtfrCJMwxwlhq91O9xr86VVp/view?usp=sharing) in `ss/checkpoints`
- Save RTE checkpint [(download)](https://drive.google.com/file/d/14InhVylKC05i2POo6gGBNXjb9EDp2Nlk/view?usp=sharing) in `rte/checkpoints`.

2. Run demo.py, i.e., `python demo.py` Make sure your gpu is available.

3. Test the model with your own claim.

## Evaluation Pipeline

    ```
    python eval_pipeline.py --dr_pipeline <id> --ss_pipeline <id> --rte_pipeline <id>
    ```

### Model pipelines
Various combinations can be implemented as followed:

    ```
    python eval_pipeline.py --dr_pipeline 2 --ss_pipeline 0 --rte_pipeline 2
    ```

1. DR
- **0**: [DocumentRetrieval] Loading wiki document titles and texts by wiki API.
- **1**: [SimpleDR] Using pre-retrieved wiki document texts to speed up the DR process
- **2**: [SimpleDR2] Using pre-retrieved wiki document titles and texts to speed up the DR process

2. SS
- **0**: [org] Trained model by using unigram similarity approach. The pipeline is loaded from `pipeline/ss_org.py`.
- **1**: [knn] K-nearest neighbors model computing the distance by the inner product of each claim and candidate sentence. The pipeline is loaded from `pipeline/ss_knn.py`

2. RTE
The pipelines are loaded from `pipeline/rte.py`
- **0**: [spe-cls] Classification model with datasets formed with single premise entailment(spe) approach. (64.67% accuracy, for test data)
- **1**: [mpe(noised)-cls] Classificaiton model with datasets formed with multiple premises entailment(mpe) approach. (76.79% accuracy, for test data)
- **2**: [mpe(noised)-rgs] Regression model with datasets formed with mpe appaorch. (67.60% accuracy, for test data)
- **3**: [spe-rgs] Regressionmodel with datasets formed with spe approach. (54.93% accuracy, for test data)

You can remove noise by adding option `--remove_noise`