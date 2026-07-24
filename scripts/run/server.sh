#!/bin/bash

# configurations
server="fedavg"
alpha=5.0
model="RESNET10"
server_port=8081
framework="torch"
dataset="CIFAR-10"
server_ip="127.0.0.1"
numClients=10
bs=64
eps=5
numClientsFit=$(($numClients/2))
execution=1000
timeout=10

server_log_path="logs/server/flwr/$server/$dataset/$alpha/$framework/$execution/$model/"
server_model_path="models/server/flwr/$server/$dataset/$alpha/$framework/$execution/$model/"
time_path_server="results/server/flwr/training/$server/$dataset/$alpha/$framework/$execution/$model/"

echo "Creating paths"
mkdir -p $server_log_path
mkdir -p $server_model_path 
mkdir -p $time_path_server

# starting server32607MiB
python -m src.federated_learning.server.$framework.app -to=$timeout -ds=$dataset -ncf=$numClientsFit -nc=$numClients -nor=$eps -sn=$server -smp=$server_model_path -md=$model -slp=$server_log_path -sp=$server_port -tp=$time_path_server -a=$alpha & 
