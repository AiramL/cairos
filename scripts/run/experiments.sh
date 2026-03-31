#!/bin/bash

framework=$(yq '.simulation.federated_learning.framework' config/config.yaml )
ip=$(yq '.simulation.federated_learning.server.ip' config/config.yaml )
port=$(yq '.simulation.federated_learning.server.port' config/config.yaml )
n_clients=$(yq '.simulation.cars' config/config.yaml )
rounds=$(yq '.simulation.federated_learning.server.rounds' config/config.yaml )
n_clients_fit=$(yq '.simulation.federated_learning.server.n_clients_fit' config/config.yaml )
model=$(yq '.simulation.federated_learning.client.model' config/config.yaml )
local_epochs=$(yq '.simulation.federated_learning.client.epochs' config/config.yaml )
dataset=$(yq '.simulation.federated_learning.client.dataset' config/config.yaml )
alpha_dirichlet=$(yq '.simulation.federated_learning.data.alpha' config/config.yaml )
timeout=$(yq '.simulation.federated_learning.server.timeout' config/config.yaml )
strategy=$(yq '.simulation.federated_learning.server.strategy' config/config.yaml )
distribution_type=$(yq '.simulation.federated_learning.server.epochs_distribution' config/config.yaml )

mapfile -t speed_ids < <(yq '.simulation.speed.index[]' config/config.yaml)

exec_id=0

echo "Starting FL training with $framework server at $ip:$port, $n_clients clients executing $local_epochs local epochs, for $rounds rounds, selecting $n_clients_fit clients to fit the $model model for $local_epochs local epochs on the dataset $dataset using a datadistribution with alpha equals to $alpha_dirichlet, a maximun timeout of $timeout seconds, using the $strategy strategy, the $distribution_type local epochs distribution, and speed index $speed_id."

for speed_id in speed_ids;
do

source scripts/run/baremetal.sh "$strategy" "$alpha_dirichlet" "$model" "$port" "$framework" "$n_clients" "$dataset" "$rounds" "$local_epochs" "$n_clients_fit" "$distribution_type" "$timeout" "$speed_id" "$exec_id"
	
done
