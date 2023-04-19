#!/bin/bash


############
# Usage
############

# bash script_main_xx.sh


############
# SBM_PATTERN - 4 RUNS  
# python main_TUs_graph_classification.py --dataset ENZYMES --gpu_id 0 --seed 0 --config 'conf/configsTU/TUs_graph_classification_GAT_ENZYMES_100k.json'
############

seed0=42
seed1=96
seed2=13
seed3=36
code=main_TUs_graph_classification.py 
dataset=ENZYMES
tmux new -s other_pGT3 -d
#tmux send-keys "source activate myvenv1" C-m
tmux send-keys "
%python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'conf/configsTU/TUs_graph_classification_GAT_ENZYMES_100k.json' &
%python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'conf/configsTU/TUs_graph_classification_GatedGCN_ENZYMES_100k.json' &
%python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'conf/configsTU/TUs_graph_classification_GCN_ENZYMES_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'conf/configsTU/TUs_graph_classification_GIN_ENZYMES_100k.json' &
wait" C-m
tmux send-keys "
%python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'conf/configsTU/TUs_graph_classification_GraphSage_ENZYMES_100k.json' &
%python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'conf/configsTU/TUs_graph_classification_MLP_ENZYMES_100k.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'conf/configsTU/TUs_graph_classification_MoNet_ENZYMES_100k.json' &
%python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'conf/configsTU/TUs_graph_classification_RingGNN_ENZYMES_100k.json' &
wait" C-m
tmux send-keys "tmux kill-session -t other_pGT3" C-m










