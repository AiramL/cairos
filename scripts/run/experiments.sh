#!/bin/bash

alpha_dirichlet="5.0"

for distribution_type in "uniform" "normal" "equal";
do

	for dataset in "CIFAR-10" "SIGN"; 
	do

		for fit in 10 20 30 40 50 60;
		do
			for local_epochs in 4 8 16;
			do
				source scripts/run/baremetal.sh "fedavg" "$alpha_dirichlet" "MOBILENETV2" "8081" "torch" "60" "$dataset" "$local_epochs" "$fit" "$distribution_type" "5"
			done
		done

	done

done
