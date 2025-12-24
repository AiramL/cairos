#!/bin/bash

speeds=$(yq '.simulation.speed.index[]' "config/config.yaml")
mobility=$(yq '.simulation.mobility.repetitions' "config/config.yaml")

for speed in $speeds

do

	for index in $( seq 0 $(($mobility-1)))

	do

		python -m utils.process.results.processed.communication $speed $index &

	done

	wait
done

wait 

echo "process finished"

