#!/bin/bash

alpha_dirichlet="5.0"
global_epochs=10
timeout=50
n_clients=10

for local_epochs in 20;
do

	#for distribution_type in "uniform" "normal" "equal";
	for distribution_type in "equal";
	do

		#for dataset in "CIFAR-10" "SIGN"; 
		for dataset in "CIFAR-10"; 
		do

			#for fit in 10 30 50;
			for fit in 10;
			do
				#for strategie in "fedavg" "cairos_pe" "cairos_pb";
				for strategie in "cairos_pe";
				do
					source scripts/run/baremetal.sh "$strategie" "$alpha_dirichlet" "RESNET10" "8081" "torch" "$n_clients" "$dataset" "$global_epochs" "$local_epochs" "$fit" "$distribution_type" "$timeout" "4"
					
				done
			done

		done

	done

done
