#!/bin/bash

echo "Creating directories"

[ ! -d datasets/VeReMi_Extension ] && mkdir -p datasets/VeReMi_Extension
[ ! -d datasets/VpnNonVpn ] && mkdir -p datasets/VpnNonVpn
[ ! -d datasets/CIFAR-10 ] && mkdir -p datasets/CIFAR-10
[ ! -d datasets/FMNIST ] && mkdir -p datasets/FMNIST
[ ! -d datasets/MNIST ] && mkdir -p datasets/MNIST
[ ! -d datasets/VNAT ] && mkdir -p datasets/VNAT
[ ! -d datasets/Modified_VeReMi/WiSec_DataModifiedVeremi_Dataset ] && mkdir -p datasets/Modified_VeReMi/WiSec_DataModifiedVeremi_Dataset

echo "Downloading dataset"

cd datasets/VeReMi_Extension
if [ ! -f "mixalldata_clean.csv" ]; then

	wget https://gta.ufrj.br/~airam/DATASETS/VeReMi/mixalldata_clean.csv --no-check-certificate

fi

cd ../VpnNonVpn
if [ ! -f "scenario_1.zip" ]; then
	
	wget -O scenario_1.zip http://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/CSVs/Scenario%20A1-ARFF.zip
	wget -O scenario_2.zip http://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/CSVs/Scenario%20A2-ARFF.zip
	wget -O scenario_3.zip http://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/CSVs/Scenario%20B-ARFF.zip

	unzip scenario_1.zip
	unzip scenario_2.zip
	unzip scenario_3.zip

fi

#cd ../VNAT
#if [ ! -f "scenario_1.zip" ]; then 
#
#	wget -O scenario_1.zip https://archive.ll.mit.edu/datasets/vnat/VNAT_release_1.zip
#	wget -O dataframe.h5 https://archive.ll.mit.edu/datasets/vnat/VNAT_Dataframe_release_1.h5
#	wget -O feature.h5 https://archive.ll.mit.edu/datasets/vnat/VNAT_Feature_Dataframe_release_1.h5
#
#fi

cd ../Modified_VeReMi/WiSec_DataModifiedVeremi_Dataset
if [ ! -f "attack1withlabels.mat" ]; then
	
	wget https://gta.ufrj.br/~airam/DATASETS/WiSec_DataModifiedVeremi_Dataset/attack1withlabels.mat --no-check-certificate
	wget https://gta.ufrj.br/~airam/DATASETS/WiSec_DataModifiedVeremi_Dataset/attack2withlabels.mat --no-check-certificate
	wget https://gta.ufrj.br/~airam/DATASETS/WiSec_DataModifiedVeremi_Dataset/attack4withlabels.mat --no-check-certificate
	wget https://gta.ufrj.br/~airam/DATASETS/WiSec_DataModifiedVeremi_Dataset/attack8withlabels.mat --no-check-certificate
	wget https://gta.ufrj.br/~airam/DATASETS/WiSec_DataModifiedVeremi_Dataset/attack16withlabels.mat --no-check-certificate
fi

cd ../../../
if [ ! -d datasets/traffic_signs ]; then
	
	mkdir -p datasets/traffic_signs
	export KAGGLEHUB_CACHE="datasets/traffic_signs"
	python utils/data/get_signs_dataset.py

fi

if [ ! -f "datasets/CIFAR-10/class0Test" ]; then
	
	cd utils/
	python data/get_image_datasets.py

fi

