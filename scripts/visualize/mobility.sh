speed=0
execution=0
cars=$(yq '.simulation.cars' "config/config.yaml")

python -m utils.process.poi

sumo-gui -n mobility/raw/scenarios/$speed/manhattan_net_2.xml -r mobility/raw/scenarios/$speed/Krauss/$cars/flows_file_Krauss_"$cars"_"$execution".xml -a mobility/map/poi.xml
