#!/bin/bash

alpha_dirichlet="5.0"
global_epochs=15
n_clients=50

for index in 9 8;
do
for fit in 10 30;
do
for local_epochs in 10;
do

	for distribution_type in "equal";
	do

		for dataset in "CIFAR-10"; 
		do
			for timeout in 50;
			do

				for strategie in "fedavg" "cairos_pe" "cairos_pb";
				do
					source scripts/run/baremetal.sh "$strategie" "$alpha_dirichlet" "RESNET10" "8081" "torch" "$n_clients" "$dataset" "$global_epochs" "$local_epochs" "$fit" "$distribution_type" "$timeout" "$timeout$index"
					
				done
			done

		done

	done

done
done
done
