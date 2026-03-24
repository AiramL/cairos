#!/bin/bash

mapfile -t values < <(yq '.simulation.speed.value[]' config/config.yaml)
mapfile -t indexes < <(yq '.simulation.speed.index[]' config/config.yaml)

for i in "${!indexes[@]}"
do

	[ ! -d mobility/processed/speed"${values[$i]}" ] && mkdir -p mobility/processed/speed"${values[$i]}"
	[ ! -d data/processed/speed"${values[$i]}" ] && mkdir -p data/processed/speed"${values[$i]}"
	[ ! -d data/raw/speed"${values[$i]}" ] && mkdir -p data/raw/speed"${values[$i]}"
	[ ! -d mobility/raw/speed"${values[$i]}" ] && mkdir -p mobility/raw/speed"${values[$i]}"

done
#[ ! -d results/client_selection/speed0 ] && mkdir -p results/client_selection/speed0
#[ ! -d results/client_selection/speed1 ] && mkdir -p results/client_selection/speed1
#[ ! -d results/client_selection/speed2 ] && mkdir -p results/client_selection/speed2
#[ ! -d results/client_selection/raw/epoch ] && mkdir -p results/client_selection/raw/epoch
#[ ! -d results/client_selection/processed ] && mkdir -p results/client_selection/processed
