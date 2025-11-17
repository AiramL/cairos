#!/bin/bash

# configurations
	
server="fedavg"
alpha=5.0
model="RESNET18"
server_port=8082
framework="torch"
cwh1=4
cwh2=0
cwh3=0
dataset="SIGN"
server_ip="146.164.69.190"
numClients=$(($cwh1+$cwh2+$cwh3))
bs=64
eps=100
numClientsFit=$(($numClients/2))
execution=0001

[ ! -d "datasets/$dataset/distributions/nclients_$numClients/alpha_$alpha/"  ] && python src/data_division/split_data.py $numClients $dataset $alpha

clients_result_path="results/clients/flwr/classification/$server/$dataset/$alpha/$framework/$execution/$model/"
clients_log_path="logs/clients/flwr/$server/$dataset/$alpha/$framework/$execution/$model/"
clients_model_path="models/clients/flwr/$server/$dataset/$alpha/$framework/$execution/$model/"
time_path_client="results/clients/flwr/training/$server/$dataset/$alpha/$framework/$execution/$model/"

mkdir -p $clients_result_path
mkdir -p $clients_result_path/raw
mkdir -p $clients_log_path
mkdir -p $clients_model_path 
mkdir -p $time_path_client

clients_result_path="results/clients/flwr/classification/$server/$dataset/$alpha/$framework/$execution/"
clients_log_path="logs/clients/flwr/$server/$dataset/$alpha/$framework/$execution/"
clients_model_path="models/clients/flwr/$server/$dataset/$alpha/$framework/$execution/"
time_path_client="results/clients/flwr/training/$server/$dataset/$alpha/$framework/$execution/"

# starting one client
cid=1
python -m src.federated_learning.client.$framework.app -ds=$dataset -es="distributed" -md=$model -nc=$numClients -cid=$cid -b=$bs -ncf=$numClientsFit -mp=$clients_model_path -lp=$clients_log_path -rp=$clients_result_path -ctp=$time_path_client -sp=$server_port -a=$alpha -sip=$server_ip >> results/clients/flwr/classification/$server/$dataset/$alpha/$framework/$execution/$model/raw/"client_""$cid" &
