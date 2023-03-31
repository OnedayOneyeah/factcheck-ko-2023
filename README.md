# factcheck-ko-2023
- **Paper**
    - [Yewon Han (2023)](https://www.notion.so/Fact-check-automation-b40c30af59a9412caff8da59e4e12921)

- **Prior projects**
    - [factcheck-ko-2021](https://github.com/hongcheki/factcheck-ko-2021)
    - [factcheck-ko-2020](https://github.com/ozmig77/factcheck-ko-2020)


## Leaderboard

|Rank|Recall(%)|SS|RTE
|---|---|---|---
|1|60.36|org|mpe(noised)-cls
|2|58.72|org|spe-cls
|3|56.16|org|mpe(noised)-rgs
|4|54.06|org|mpe-rgs
|5|52.58|org|spe-rgs
|6|50.88|org|mpe-cls
|7|48.82|KNN(k=5)|spe-cls
|8|43.54|KNN(k=5)|mpe(noised)-cls
|9|40.01|KNN(k=5)|mpe(noised)-rgs
|10|37.72|KNN(k=5)|mpe-cls

---
## Install
```
git clone https://github.com/hongcheki/factcheck-ko-2021.git
cd factcheck-ko-2021
conda create -n factcheck-ko-2021 python=3.8
conda activate factcheck-ko-2021
pip install -r requirements.txt
```

---
## Data

Download the data for training [here](https://drive.google.com/drive/folders/1cYJejZ6gxT7TARy7BtWN77384VgYmjoE?usp=sharing). Save the files in `./data/`
- `data/wiki_claims.json`: Human-Annotated Dataset for the Factcheck
- `data/train_val_test_ids.json`: Lists of claim ids for train/validation/test split

---
## Train/Test
### SS/RTE model
- Documents are available [here](https://github.com/hongcheki/factcheck-ko-2021).
- Save SS checkpoint [(download)](https://drive.google.com/file/d/1-XuWTl2PKtfrCJMwxwlhq91O9xr86VVp/view?usp=sharing) in `ss/checkpoints`
- Save RTE checkpoint [(download)](https://drive.google.com/file/d/14InhVylKC05i2POo6gGBNXjb9EDp2Nlk/view?usp=sharing) in `rte/checkpoints`
- Save additional RTE checkpoints in `new_rte/checkpoints`: [(spe-rgs)](https://drive.google.com/file/d/1Qsd1Aq3daLndqh5r03dc3d3bE1vhsGIr/view?usp=sharing), [(mpe-rgs)](https://drive.google.com/file/d/1Pgf2LHp_IQfyCaViB44O91PAwx4Xdras/view?usp=sharing), [(mpe-cls)](https://drive.google.com/file/d/1QrYEMGfZL0MxkcytREvgB9damWiAAlix/view?usp=sharing)
    

    > **[Descriptions]**  
        - ***spe***: single premise entailment approach  
        - ***mpe***: multiple premises entailment approach  
        - ***cls***: classifcation model  
        - ***rgs***: regression model  


### Fact-check model

#### Demo
1. Make sure you download the pretrained checkpoints for SS, RTE model.

2. Run demo.py, i.e., `python demo.py` Make sure your gpu is available.

3. Test the model with your own claim.


#### Evaluation pipeline
- For evaluation, run eval_pipeline.py, i.e., `eval_pipeline.py`
    ```
    python eval_pipeline.py --dr_pipeline <id> --ss_pipeline <id> --rte_pipeline <id>
    ```
- Various combinations can be implemented as followed:

    ```
    python eval_pipeline.py --dr_pipeline 2 --ss_pipeline 0 --rte_pipeline 2
    ```


#### Model pipelines


**1. DR**
- *The pipelines are loaded from `dr/document_retrieval.py` or `eval_pipeline.py`*
- `SimpleDR` and `SimpleDR2` reduce time for document retrieval.

    |Id|Model|Description|
    |---|---|---|
    |0|DocumentRetrieval|Loading wiki document titles and texts using wiki API|
    |1|SimpleDR|Using pre-retrieved wiki document texts|
    |2|SimpleDR2|Using pre-retrieved wiki document titles and texts|


**2. SS**
- *The pipelines are loaded from `pipelines/ss_org.py` and `pipelines/ss_knn.py` respectively.*

  |Id|Model|Description|
  |---|---|---|
  |0|org|Unigram similarity approach|
  |1|knn|K-nearest neighbors apporach|


**3. RTE**
- *The pipelines are loaded from `pipelines/rte.py`*

    |Id|Model|Recall(%)|
    |---|---|---|
    |0|spe-cls|64.67|
    |1|mpe(noised)-cls|76.79|
    |2|mpe(noised)-rgs|67.60|
    |3|spe-rgs|54.93|


- You can remove noise by `--remove_noise` option.