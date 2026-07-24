#!/bin/bash

# configurations
	
server="cairos_pe"
alpha=5.0
model="RESNET10"
server_port=8081
framework="torch"
dataset="CIFAR-10"
server_ip="127.0.0.1"
numClients=10
bs=64
eps=3
data_type="delay"
numClientsFit=$(($numClients/2))
execution=1000
base_station_range=10000
timeout=10
i_epochs=10
speed_id=0

[ ! -d "datasets/$dataset/distributions/nclients_$numClients/alpha_$alpha/"  ] && python -m src.data_division.split_data $numClients $dataset $alpha

clients_result_path="results/clients/flwr/classification/$server/$dataset/$alpha/$framework/$timeout/$i_epochs/$numClientsFit/$scenario/$data_type/$execution/$model/"
clients_log_path="logs/clients/flwr/$server/$dataset/$alpha/$framework/$timeout/$i_epochs/$numClientsFit/$scenario/$data_type/$execution/$model/"
clients_model_path="models/clients/flwr/$server/$dataset/$alpha/$framework/$timeout/$i_epochs/$numClientsFit/$scenario/$data_type/$execution/$model/"
time_path_client="results/clients/flwr/training/$server/$dataset/$alpha/$framework/$timeout/$i_epochs/$numClientsFit/$scenario/$data_type/$execution/$model/"

mkdir -p $clients_result_path
mkdir -p $clients_result_path/raw
mkdir -p $clients_log_path
mkdir -p $clients_model_path 
mkdir -p $time_path_client

clients_result_path="results/clients/flwr/classification/$server/$dataset/$alpha/$framework/$timeout/$i_epochs/$numClientsFit/$scenario/$data_type/$execution/"
clients_log_path="logs/clients/flwr/$server/$dataset/$alpha/$framework/$timeout/$i_epochs/$numClientsFit/$scenario/$data_type/$execution/"
clients_model_path="models/clients/flwr/$server/$dataset/$alpha/$framework/$timeout/$i_epochs/$numClientsFit/$scenario/$data_type/$execution/"
time_path_client="results/clients/flwr/training/$server/$dataset/$alpha/$framework/$timeout/$i_epochs/$numClientsFit/$scenario/$data_type/$execution/"

# starting one client
if [ $# -eq 0 ]
then
    echo "No arguments supplied, running client 0"
    cid=0
else
    cid=$1
fi

for i in $(seq 1 $(($numClients-1)));
do
    
        echo "Waiting client "$i" initialization"
        python -m src.federated_learning.client.$framework.$data_type.app -epb=0 -ot=0 -eid=0 -spid=$speed_id -mt=$timeout -nle=$i_epochs -ds=$dataset -md=$model -nc=$numClients -cid=$i -b=$bs -ncf=$numClientsFit -mp=$clients_model_path -lp=$clients_log_path -rp=$clients_result_path -ctp=$time_path_client -sp=$server_port -a=$alpha -bsr=$base_station_range >> $clients_result_path$model"/raw/client_$i" &
        
        sleep 1

done


		