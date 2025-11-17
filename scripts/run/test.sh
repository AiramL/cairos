#!/bin/bash

epoch="5" 
temperature="10"
alp="0.5"

for dataset in "SIGN";
do

	for alpha_dirichlet in 5.0;
	do 


		source scripts/run/baremetal.sh "fhemda" "$alpha_dirichlet" "RESNET18" "8097" "torch" "1" "1" "2"   "$alp" "$temperature" "$epoch" "$dataset" "22"
		#source scripts/run/baremetal.sh "fedavg_cwh1" "$alpha_dirichlet" "RESNET18" "8097" "torch" "4" "0" "0"   "$alp" "$temperature" "$epoch" "$dataset" "22"
		#source scripts/run/baremetal.sh "fedavg_cwh2" "$alpha_dirichlet" "RESNET10" "8097" "torch" "0" "4" "0"   "$alp" "$temperature" "$epoch" "$dataset" "22"
		#source scripts/run/baremetal.sh "fedavg_cwh3" "$alpha_dirichlet" "CNN" "8097" "torch" "0" "0" "4"   "$alp" "$temperature" "$epoch" "$dataset" "22"


	done

done
