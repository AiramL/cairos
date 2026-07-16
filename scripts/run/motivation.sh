#!/bin/bash

alpha_dirichlet="5.0"
global_epochs=15
n_clients=50


for fit in 1 25 50 10;
do
for local_epochs in 5;
do

	for distribution_type in "equal";
	do

		for dataset in "CIFAR-10"; 
		do
			for timeout in 2000;
			do

				for strategie in "fedavg";
				do
					source scripts/run/baremetal.sh "$strategie" "$alpha_dirichlet" "RESNET10" "8081" "torch" "$n_clients" "$dataset" "$global_epochs" "$local_epochs" "$fit" "$distribution_type" "$timeout" "$timeout"
					
				done
			done

		done

	done

done
done
