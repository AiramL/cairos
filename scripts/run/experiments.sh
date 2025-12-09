#!/bin/bash

alpha_dirichlet="5.0"

for distribution_type in "uniform" "poison" "normal" "equal";
do

	for dataset in "CIFAR-10" "SIGN"; 
	do

		for fit in 1 2 4 8 16 20;
		do
			for local_epochs in 1 2 4 8 16;
			do
				source scripts/run/baremetal.sh "fedavg" "$alpha_dirichlet" "MOBILENETV2" "8081" "torch" "20" "$dataset" "$local_epochs" "$fit" "$distribution_type" "3"
			done
		done

	done

done
