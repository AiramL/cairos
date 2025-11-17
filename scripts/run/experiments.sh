#!/bin/bash

alpha_dirichlet="5.0"

for dataset in "SIGN" "CIFAR-10"; 
do

	for local_epochs in 1 2 4 8 16 32;
	do
		source scripts/run/baremetal.sh "fedavg" "$alpha_dirichlet" "RESNET10" "8081" "torch" "20" "$dataset" "$local_epochs" "1"
	done

done
