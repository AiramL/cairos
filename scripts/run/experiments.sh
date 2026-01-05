#!/bin/bash

alpha_dirichlet="5.0"
global_epochs=5
timeout=120
n_clients=4

for strategie in "fedavg" "cairos";
do

	#for distribution_type in "uniform" "normal" "equal";
	for distribution_type in "equal";
	do

		#for dataset in "CIFAR-10" "SIGN"; 
		for dataset in "CIFAR-10"; 
		do

			#for fit in 10 20 30 40 50;
			for fit in 2;
			do
				for local_epochs in 10;
				do
					source scripts/run/baremetal.sh "$strategie" "$alpha_dirichlet" "RESNET10" "8081" "torch" "$n_clients" "$dataset" "$global_epochs" "$local_epochs" "$fit" "$distribution_type" "$timeout" "1"
					
				done
			done

		done

	done

done
