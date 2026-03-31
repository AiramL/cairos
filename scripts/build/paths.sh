#!/bin/bash

for i in 0 1 2;
do

	[ ! -d mobility/processed/speed"$i" ] && mkdir -p mobility/processed/speed"$i"
	[ ! -d data/processed/speed"$i" ] && mkdir -p data/processed/speed"$i"
	[ ! -d data/raw/speed"$i" ] && mkdir -p data/raw/speed"$i"
	[ ! -d mobility/raw/speed"$i" ] && mkdir -p mobility/raw/speed"$i"

done
	
[ ! -d models/ ] && mkdir -p models/
