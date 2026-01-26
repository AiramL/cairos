#!/bin/bash

alpha_dirichlet="5.0"
global_epochs=20
timeout=50
n_clients=10

for local_epochs in 10 5 20;
do

	#for distribution_type in "uniform" "normal" "equal";
	for distribution_type in "equal";
	do

		for dataset in "CIFAR-10" "SIGN"; 
		do

			for fit in 10 30 50;
			do
				for strategie in "cairos_pe" "cairos_pb";
				do
					source scripts/run/baremetal.sh "$strategie" "$alpha_dirichlet" "RESNET10" "8081" "torch" "$n_clients" "$dataset" "$global_epochs" "$local_epochs" "$fit" "$distribution_type" "$timeout" "5"
					
				done
			done

		done

	done

done
