# Using Graph Neural Network to Predict CS Paper's Subject Area
This repository contains my attempts at predicting the subject area (e.g) of computer science papers on Arxiv, using [dgl](https://github.com/dmlc/dgl/)'s implementation of Graph Convolutional Neural Networks and Graph Attention Networks. Models are trained using the [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework.

Dataset is from the [Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv).

To run the training script:
```python
  mkdir weights
  mkdir logs
  python3 train_lightning.py
```

Note that the script does not use any GPU (since my local setup does not have any GPU). However, switching to using GPU is easy; simply add `gpus=1` arguments to `Trainer` initialization.

Colab Notebooks:
  * [Simple Baseline (MLP)](https://colab.research.google.com/drive/15fPSGUzZI0BFIXgKdGNgyLDABd0je0JX?usp=sharing)
  * [Two Layers Graph Convolutional Neural Network](https://colab.research.google.com/drive/12CQ4rsbW2vpyUn4Wu4lQ9trusidwnyuD?usp=sharing)
