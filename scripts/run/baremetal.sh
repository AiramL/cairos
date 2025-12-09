#!/bin/bash

if [ -z $1 ]; then
	
	server="fedavg"
	alpha=0.1
	model="RESNET18"
	server_port=8082
	framework="keras"
	numClients=20
	dataset="SIGN"
	i_epochs="5"
	numClientsFit=$(($numClients/2))
	scenario="uniform"
	execution="1"

else
	
	server=$1
	alpha=$2
	model=$3
	server_port=$4
	framework=$5
	numClients=$6
	dataset=$7
	i_epochs=$8
	numClientsFit=$9
	scenario=${10}
	execution=${11}

fi

bs=64
eps=50

server_log_path="logs/server/flwr/$server/$dataset/$alpha/$framework/$execution/$i_epochs/$numClientsFit/$scenario/$model/"
server_model_path="models/server/flwr/$server/$dataset/$alpha/$framework/$execution/$i_epochs/$numClientsFit/$scenario/$model/"
time_path_server="results/server/flwr/training/$server/$dataset/$alpha/$framework/$execution/$i_epochs/$numClientsFit/$scenario/$model/"

echo "Creating paths"
mkdir -p $server_log_path
mkdir -p $server_model_path 
mkdir -p $time_path_server

epochs_list=$(python -m utils.epochs_distributions $numClients $i_epochs $scenario)
epochs_array=($epochs_list)

[ ! -d "datasets/$dataset/distributions/nclients_$numClients/alpha_$alpha/"  ] && python src/data_division/split_data.py $numClients $dataset $alpha


if [ $server == "fedavg" ]; then

	echo "Starting server fedavg"
	sleep 3
	python -m src.federated_learning.server.$framework.app -ds=$dataset -ncf=$numClientsFit -nc=$numClients -nor=$eps -sn=$server -smp=$server_model_path -md=$model -slp=$server_log_path -sp=$server_port -tp=$time_path_server -a=$alpha & 
		
	clients_result_path="results/clients/flwr/classification/$server/$dataset/$alpha/$framework/$execution/$i_epochs/$numClientsFit/$scenario/$model/"
	clients_log_path="logs/clients/flwr/$server/$dataset/$alpha/$framework/$execution/$i_epochs/$numClientsFit/$scenario/$model/"
	clients_model_path="models/clients/flwr/$server/$dataset/$alpha/$framework/$execution/$i_epochs/$numClientsFit/$scenario/$model/"
	time_path_client="results/clients/flwr/training/$server/$dataset/$alpha/$framework/$execution/$i_epochs/$numClientsFit/$scenario/$model/"

	mkdir -p $clients_result_path
	mkdir -p $clients_result_path/raw
	mkdir -p $clients_log_path
	mkdir -p $clients_model_path 
	mkdir -p $time_path_client
	
	clients_result_path="results/clients/flwr/classification/$server/$dataset/$alpha/$framework/$execution/$i_epochs/$numClientsFit/$scenario/"
	clients_log_path="logs/clients/flwr/$server/$dataset/$alpha/$framework/$execution/$i_epochs/$numClientsFit/$scenario/"
	clients_model_path="models/clients/flwr/$server/$dataset/$alpha/$framework/$execution/$i_epochs/$numClientsFit/$scenario/"
	time_path_client="results/clients/flwr/training/$server/$dataset/$alpha/$framework/$execution/$i_epochs/$numClientsFit/$scenario/"
		
	echo "Starting clients fedavg"
	sleep 10
	
	for i in $(seq 0 $(($numClients-1)))
	do
	
		echo "Waiting client "$i" initialization"
		python -m src.federated_learning.client.$framework.app -nle="${epochs_array[$i]}" -ds=$dataset -md=$model -nc=$numClients -cid=$i -b=$bs -ncf=$numClientsFit -mp=$clients_model_path -lp=$clients_log_path -rp=$clients_result_path -ctp=$time_path_client -sp=$server_port -a=$alpha >> results/clients/flwr/classification/$server/$dataset/$alpha_dirichlet/$framework/$execution/$i_epochs/$numClientsFit/$scenario/$model/raw/"client_""$i" &
		
		sleep 1

	done
fi

wait

echo "baremetal script finished"
