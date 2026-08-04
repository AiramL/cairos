#!/bin/bash

framework=$(yq '.simulation.federated_learning.framework' config/config.yaml )
framework=$(echo "$framework" | tr -d '"')

port=$(yq '.simulation.federated_learning.server.port' config/config.yaml )
port=$(echo "$port" | tr -d '"')

n_clients=$(yq '.simulation.cars' config/config.yaml )
n_clients=$(echo "$n_clients" | tr -d '"')

rounds=$(yq '.simulation.federated_learning.server.rounds' config/config.yaml )
rounds=$(echo "$rounds" | tr -d '"')

base_station_range=$(yq '.simulation.base_station.range' config/config.yaml ) 
base_station_range=$(echo "$base_station_range" | tr -d '"')

fit=$(yq '.simulation.federated_learning.server.n_clients_fit' config/config.yaml )
fit=$(echo "$fit" | tr -d '"')

model=$(yq '.simulation.federated_learning.client.model' config/config.yaml )
model=$(echo "$model" | tr -d '"')

distribution_type=$(yq '.simulation.federated_learning.server.epochs_distribution' config/config.yaml )
distribution_type=$(echo "$distribution_type" | tr -d '"')

speed_id=0
data_type="delay"
data_type=$(echo "$data_type" | tr -d '"')

for dataset in "CIFAR-10" "SIGN"; 
do

for alpha_dirichlet in 0.5 1.0 5.0;
do

for local_epochs in 10 5 20;
do

for fit in 1 2 5 10 20 40;
do

for timeout in 1000;
do

for index in 0 1 2; 
do

for strategie in "fedavg";
do

source scripts/run/baremetal.sh "$strategie" "$alpha_dirichlet" "$model" "$port" "$framework" "$n_clients" "$dataset" "$rounds" "$local_epochs" "$fit" "$distribution_type" "$timeout" "$speed_id" "$index" "$base_station_range" "$data_type"
					
done
done
done
done
done
done
done
