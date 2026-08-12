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

local_epochs=$(yq '.simulation.federated_learning.client.epochs' config/config.yaml )
local_epochs=$(echo "$local_epochs" | tr -d '"')

distribution_type=$(yq '.simulation.federated_learning.server.epochs_distribution' config/config.yaml )
distribution_type=$(echo "$distribution_type" | tr -d '"')

mapfile -t speed_ids < <(yq '.simulation.speed.index[]' config/config.yaml)

for dataset in "CIFAR-10" "SIGN"; 
do

for alpha_dirichlet in 0.5 5.0;
do

for data_type in "delay";
do

for speed_id in $speed_ids;
do

speed_ids=$(echo "$speed_ids" | tr -d '"')

for timeout in 5 10 20 50;
do

for strategie in "fedavg" "cairos_pe" "cairos_pb";
do

for index in 0 1 2;
do

source scripts/run/baremetal.sh "$strategie" "$alpha_dirichlet" "$model" "$port" "$framework" "$n_clients" "$dataset" "$rounds" "$local_epochs" "$fit" "$distribution_type" "$timeout" "$speed_id" "$index" "$base_station_range" "$data_type"

done
done
done
done
done
done
done
