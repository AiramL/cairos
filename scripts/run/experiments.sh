#!/bin/bash

alpha_dirichlet="5.0"
global_epochs=50
timeout=50
n_clients=50

for local_epochs in 10 5 20;
do

	#for distribution_type in "uniform" "normal" "equal";
	for distribution_type in "equal";
	do

		for dataset in "CIFAR-10" "SIGN"; 
		do

			for fit in 10 20 30 40 50;
			do
				#for strategie in "fedavg" "cairos";
				for strategie in "cairos";
				do
					source scripts/run/baremetal.sh "$strategie" "$alpha_dirichlet" "RESNET10" "8081" "torch" "$n_clients" "$dataset" "$global_epochs" "$local_epochs" "$fit" "$distribution_type" "$timeout" "2"
					
				done
			done

		done

	done

done
