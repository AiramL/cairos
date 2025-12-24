# This code executes the federated learning training, starting clients and server
#                                            eps  tofl  ncl ncf  bs   strategy dataset
#source scripts/run/all_clients_baremetal.sh "40"  "0"  "100" "16" "128" "random" "VeReMi" & 
#wait
#source scripts/run/all_clients_baremetal.sh "40"  "0"  "100" "16" "128" "random" "WiSec" & 
#wait
#source scripts/run/all_clients_baremetal.sh "40"  "0"  "100" "16" "128" "m_fastest" "VeReMi" & 
#wait
#source scripts/run/all_clients_baremetal.sh "40"  "0"  "100" "16" "128" "m_fastest" "WiSec" & 
#wait
source scripts/run/all_clients_baremetal.sh "40"  "0"  "100" "95" "128" "random" "VeReMi" & 
wait
source scripts/run/all_clients_baremetal.sh "40"  "0"  "100" "95" "128" "random" "WiSec" & 
wait
source scripts/run/all_clients_baremetal.sh "40"  "0"  "100" "95" "128" "m_fastest" "VeReMi" & 
wait
source scripts/run/all_clients_baremetal.sh "40"  "0"  "100" "95" "128" "m_fastest" "WiSec" & 
wait
