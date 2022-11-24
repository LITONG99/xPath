# xPath
This is a [PyTorch](https://pytorch.org/) implementation for our AAAI'23 paper: Towards Fine-grained Explainability for Heterogeneous Graph Neural Network.

Here also include the supplementary materials for proof of theorems in the paper and additional experimental results.

An alternative implementation based on [Mindspore](https://www.mindspore.cn/) is coming soon.


## Requirements
- Python 3.7
- torch~=1.10.0
- dgl~=0.8.2
- numpy~=1.21.6
- tqdm~=4.64.0


## Datasets

The heterogeneous graph datasets we use are [DBLP](https://github.com/BUPT-GAMMA/HeCo/tree/main/data/dblp), [ACM](https://github.com/BUPT-GAMMA/HeCo/tree/main/data/acm), and [IMDB](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset). 
Datasets are provided in the `data` folder. Taking DBLP dataset as an example:
- `dblp_graph.bin`: Heterogeneous graph。
- `dblp_index_60.bin`: Training, validation, and test set for SimpleHGN.
- `dblp_index_2000.bin`: Training, validation, and test set for HGT.

## Train HGNs

Trained HGNs are provided in `ckpt/{dataset_name}/bk` for reproducing the results in our paper. To retrain the HGNs, run

```shell
# edit configurations in config.py
python train.py
```

## Generate explanations

```shell
# edit configurations in config.py
python main.py
```

