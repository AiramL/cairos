#!/bin/bash

server_port=$1

processes=$(ps aux | grep $server_port | grep -v grep | awk '{print $2}')

for process in $process
do
	kill -9  $process
done
