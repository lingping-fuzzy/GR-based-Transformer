# GR-representation Transformer Architecture

Source code for the paper "**[Low-Rank and Global-Representation-Key-based Attention for Graph Transformer]([link to paper](https://www.sciencedirect.com/science/article/pii/S002002552300693X?dgcid=coauthor))**".

Low-Rank and Global-Representation-Key-based Attention : 
- We proposed a low-rank GR-Key-based attention mechanism, where the global representation(s) forms the \textit{Key} providing the structural importance of neighbor nodes. This proposed attention method decreases the training parameter number and is adjustable to the latent feature rank. 
- We propose investigating various compositions of virtual representation(s) that explore the isomorphic and heterogeneous properties in graph data, which alleviates the over-smoothing and heterophily issues.
- We incorporate the proposed attention mechanism into a simple GNN model, accomplish regression/classification tasks on eight widely used benchmark datasets, and justify the proposed method's effectiveness and superiority. 
<br>

<p align="center">
  <img src="./docs/transf.png" alt="GR" width="300">
  <br>
  <b>Figure</b>: Block Diagram of GR-based Transformer Architecture
</p>

<p align="center">
  <img src="./docs/transfedge.png" alt="GR" width="300">
  <br>
  <b>Figure</b>: Block Diagram of GR-based Transformer Architecture
</p>

## https://chrsmrrs.github.io/datasets/docs/datasets/
## 1. Repo installation

This project is based on the [benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns) repository.
and [Vijay Prakash Dwivedi](https://github.com/vijaydwivedi75), [graphdeeplearning/graphtransformer](https://github.com/graphdeeplearning/graphtransformer) and [Xavier Bresson](https://github.com/xbresson)_
[Follow these instructions](./docs/01_benchmark_installation.md) to install the benchmark and setup the environment.


<br>

## 2. Download datasets and 3. Reproducibility 

please read the docs

<br>

## 4. Reference 
 
```
[1] [graphTransformer](https://arxiv.org/abs/2012.09699)
```
## Citation
If you find our work relevant to your research, please cite:
```
@article{kong2023low,
  title={Low-rank and global-representation-key-based attention for graph transformer},
  author={Kong, Lingping and Ojha, Varun and Gao, Ruobin and Suganthan, Ponnuthurai Nagaratnam and Sn{\'a}{\v{s}}el, V{\'a}clav},
  journal={Information Sciences},
  volume={642},
  pages={119108},
  year={2023},
  publisher={Elsevier}
}
```

<br><br><br>

