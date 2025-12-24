#!/bin/bash

[ ! -d data/processed/speed0 ] && mkdir -p data/processed/speed0
[ ! -d data/processed/speed1 ] && mkdir -p data/processed/speed1
[ ! -d data/processed/speed2 ] && mkdir -p data/processed/speed2
[ ! -d data/raw/speed0 ] && mkdir -p data/raw/speed0
[ ! -d data/raw/speed1 ] && mkdir -p data/raw/speed1
[ ! -d data/raw/speed2 ] && mkdir -p data/raw/speed2
[ ! -d mobility/processed/speed0 ] && mkdir -p mobility/processed/speed0
[ ! -d mobility/processed/speed1 ] && mkdir -p mobility/processed/speed1
[ ! -d mobility/processed/speed2 ] && mkdir -p mobility/processed/speed2
[ ! -d mobility/raw/speed0 ] && mkdir -p mobility/raw/speed0
[ ! -d mobility/raw/speed1 ] && mkdir -p mobility/raw/speed1
[ ! -d mobility/raw/speed2 ] && mkdir -p mobility/raw/speed2
[ ! -d results/client_selection/speed0 ] && mkdir -p results/client_selection/speed0
[ ! -d results/client_selection/speed1 ] && mkdir -p results/client_selection/speed1
[ ! -d results/client_selection/speed2 ] && mkdir -p results/client_selection/speed2
[ ! -d results/client_selection/raw/epoch ] && mkdir -p results/client_selection/raw/epoch
[ ! -d results/client_selection/processed ] && mkdir -p results/client_selection/processed
[ ! -d results/selected_clients ] && mkdir -p results/selected_clients
