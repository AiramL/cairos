#!/bin/bash

framework=$(yq '.simulation.federated_learning.framework' config/config.yaml )
port=$(yq '.simulation.federated_learning.server.port' config/config.yaml )
n_clients=$(yq '.simulation.cars' config/config.yaml )
rounds=$(yq '.simulation.federated_learning.server.rounds' config/config.yaml )
alpha_dirichlet=$(yq '.simulation.federated_learning.data.alpha' config/config.yaml )
base_station_range=$(yq '.simulation.base_station.range' config/config.yaml ) 
fit=$(yq '.simulation.federated_learning.server.n_clients_fit' config/config.yaml )
model=$(yq '.simulation.federated_learning.client.model' config/config.yaml )
local_epochs=$(yq '.simulation.federated_learning.client.epochs' config/config.yaml )
distribution_type=$(yq '.simulation.federated_learning.server.epochs_distribution' config/config.yaml )
data_type=$(yq '.simulation.data_type' config/config.yaml )

mapfile -t speed_ids < <(yq '.simulation.speed.index[]' config/config.yaml)


for speed_id in $speed_ids;
do

for index in 0 1 2;
do

for dataset in "CIFAR-10" "SIGN"; 
do

for timeout in 10 20 50 100;
do

for strategie in "fedavg" "cairos_pe" "cairos_pb";
do
source scripts/run/baremetal.sh "$strategie" "$alpha_dirichlet" "$model" "$port" "$framework" "$n_clients" "$dataset" "$rounds" "$local_epochs" "$fit" "$distribution_type" "$timeout" "$speed_id" "$index" "$base_station_range" "$data_type"

done
done
done
done
done
