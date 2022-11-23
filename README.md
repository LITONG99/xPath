# xPath
This is a [PyTorch](https://pytorch.org/) implementation for our AAAI'23 paper: Towards Fine-grained Explainability for Heterogeneous Graph Neural Network.

Here also include the supplementary materials for proof of theorems in the paper and additional experimental results.

An alternative implementation based on [Mindspore](https://www.mindspore.cn/) is coming soon.


## Requirements
- Python 3.7
- See requirements.txt for more details.

## Datasets

The heterogeneous graph datasets we use are [DBLP](https://github.com/BUPT-GAMMA/HeCo/tree/main/data/dblp), [ACM](https://github.com/BUPT-GAMMA/HeCo/tree/main/data/acm), and [IMDB](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset). 
We put the processed data sets in the `ckpt` folder. Taking DBLP dataset as an example, corresponding graph data include:
- `dblp_graph.bin`: Heterogeneous graph。
- `dblp_index_60.bin`: Training, validation, and test set for SimpleHGN.
- `dblp_index_2000.bin`: Training, validation, and test set for HGT.

## Train HGNs

We provide the trained HGNs in `ckpt/{dataset_name}/bk` for reproducing the results in our paper. To retrain the HGNs, run

```shell
# edit configurations in config.py
python train.py
```

## Generate explanations

```shell
# edit configurations in config.py
python main.py
```
