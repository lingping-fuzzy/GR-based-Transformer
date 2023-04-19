# Reproducibility


<br>

## 1. Usage
docs \
https://docs.dgl.ai/en/0.6.x/api/python/dgl.DGLGraph.html

<br>

### In terminal

```
# Run the main file (at the root of the project)
python main_molecules_graph_regression.py --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json' # for CPU
python main_molecules_graph_regression.py --gpu_id 0 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json' # for GPU
```
The training and network parameters for each experiment is stored in a json file in the [`configs/`](../configs) directory.




<br>

## 2. Output, checkpoints and visualizations

Output results are located in the folder defined by the variable `out_dir` in the corresponding config file (eg. [`configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json`](../configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json) file).  

If `out_dir = 'out/ZINC_sparse_LapPE_BN/'`, then 

#### 2.1 To see checkpoints and results
1. Go to`out/ZINC_sparse_LapPE_BN/results` to view all result text files.
2. Directory `out/ZINC_sparse_LapPE_BN/checkpoints` contains model checkpoints.

#### 2.2 To see the training logs in Tensorboard on local machine
1. Go to the logs directory, i.e. `out/ZINC_sparse_LapPE_BN/logs/`.
2. Run the commands
```
source activate graph_transformer
tensorboard --logdir='./' --port 6006
```
```
tensorboard --logdir=./out/ZINC_sparse_LapPE_BN/logs/
```

3. Open `http://localhost:6006` in your browser. Note that the port information (here 6006 but it may change) appears on the terminal immediately after starting tensorboard.


#### 2.3 To see the training logs in Tensorboard on remote machine
1. Go to the logs directory, i.e. `out/ZINC_sparse_LapPE_BN/logs/`.
2. Run the [script](../scripts/TensorBoard/script_tensorboard.sh) with `bash script_tensorboard.sh`.
3. On your local machine, run the command `ssh -N -f -L localhost:6006:localhost:6006 user@xx.xx.xx.xx`.
4. Open `http://localhost:6006` in your browser. Note that `user@xx.xx.xx.xx` corresponds to your user login and the IP of the remote machine.



<br>

## 3. Reproduce results 


```
# At the root of the project 

# reproduce main results (Table 1 in paper) 
bash scripts/ZINC/script_main_molecules_graph_regression_ZINC_500k.sh 
bash scripts/SBMs/script_main_SBMs_node_classification_CLUSTER_500k.sh 
bash scripts/SBMs/script_main_SBMs_node_classification_PATTERN_500k.sh

# reproduce WL-PE ablation results (Table 3 in paper)
bash scripts/ZINC/script_main_molecules_graph_regression_ZINC_500k_WL_ablation.sh 
bash scripts/SBMs/script_main_SBMs_node_classification_CLUSTER_500k_WL_ablation.sh
bash scripts/SBMs/script_main_SBMs_node_classification_PATTERN_500k_WL_ablation.sh
```

Scripts are [located](../scripts/) at the `scripts/` directory of the repository.

 

 <br>

## 4. Generate statistics obtained over mulitple runs 
After running a script, statistics (mean and standard variation) can be generated from a notebook. For example, after running the script `scripts/ZINC/script_main_molecules_graph_regression_ZINC_500k.sh`, go to the results folder `out/ZINC_sparse_LapPE_LN/results/`, and run the [notebook](../scripts/StatisticalResults/generate_statistics_molecules_graph_regression_ZINC.ipynb) `scripts/StatisticalResults/generate_statistics_molecules_graph_regression_ZINC.ipynb` to generate the statistics.




## 5. temporal a look of model
some of the resuls

MODEL/Total parameters: GraphTransformer 567873\
Training Graphs:  10000\
Validation Graphs:  1000\
Test Graphs:  1000
Epoch 0:   0%|          | 0/2 [00:00<?, ?it/s]D:\software\javaR\python\lib\site-packages\torch\autocast_mode.py:141: UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling
  warnings.warn('User provided device_type of \'cuda\', but CUDA is not available. Disabling')\
Epoch 1: 100%|██████████| 2/2 [09:18<00:00, 279.09s/it]\
Test MAE: 0.7908\
Train MAE: 0.7510\
Convergence Time (Epochs): 1.0000\
TOTAL TIME TAKEN: 740.3141s\
AVG TIME PER EPOCH: 279.0880s\

20 epoch\
MODEL/Total parameters: GraphTransformer 567873\

Epoch 0:   0%|          | 0/20 [00:00<?, ?it/s]D:\software\javaR\python\lib\site-packages\torch\autocast_mode.py:141: UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling
  warnings.warn('User provided device_type of \'cuda\', but CUDA is not available. Disabling')
Epoch 19: 100%|██████████| 20/20 [1:34:24<00:00, 283.21s/it]\
Test MAE: 0.5567\
Train MAE: 0.4997\
Convergence Time (Epochs): 19.0000\
TOTAL TIME TAKEN: 5875.7259s\
AVG TIME PER EPOCH: 283.2035s\

Process finished with exit code 0

----------------
the following use edge-original\
Using backend: pytorch\
cuda not available\
[I] Loading dataset ZINC...\
train, test, val sizes : 10000 1000 1000\
[I] Finished loading.\
[I] Data load time: 4.7008s\
MODEL DETAILS:\

MODEL/Total parameters: GraphTransformer 588353\
Training Graphs:  10000\
Validation Graphs:  1000\
Test Graphs:  1000\
Epoch 0:   0%|          | 0/2 [00:00<?, ?it/s]D:\software\javaR\python\lib\site-packages\torch\autocast_mode.py:141: UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling
  warnings.warn('User provided device_type of \'cuda\', but CUDA is not available. Disabling')\
Epoch 1: 100%|██████████| 2/2 [02:57<00:00, 88.60s/it]\
Test MAE: 0.6072\
Train MAE: 0.5617\
Convergence Time (Epochs): 1.0000\
TOTAL TIME TAKEN: 216.9244s\
AVG TIME PER EPOCH: 88.6001s\

20epoc\
Epoch 0:   0%|          | 0/20 [00:00<?, ?it/s]D:\software\javaR\python\lib\site-packages\torch\autocast_mode.py:141: UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling
  warnings.warn('User provided device_type of \'cuda\', but CUDA is not available. Disabling')
Epoch 19: 100%|██████████| 20/20 [29:47<00:00, 89.37s/it]\
Test MAE: 0.4763\
Train MAE: 0.4194\
Convergence Time (Epochs): 19.0000\
TOTAL TIME TAKEN: 1827.2184s\
AVG TIME PER EPOCH: 89.3649s\

Process finished with exit code 0


Process finished with exit code 0



### 6. some packages version
-> pytorch=1.10.0
- torchvision==0.7.0
-> pillow==8.3.2
- dgl=0.6.1
-> numpy=1.21.2
-> matplotlib=3.4.3
-> tensorboard=2.7.0
-> tensorboardx=2.4.1
-> future=0.18.2
- absl-py
-> networkx=2.6.3
-> scikit-learn=1.0.2
-> scipy=1.7.3










<br><br><br>
