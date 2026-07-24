data_type=$(yq '.simulation.data_type' config/config.yaml )
data_type=$(echo "$data_type" | tr -d '"')

speeds=$(yq '.simulation.speed.index[]' "config/config.yaml")

for speed in $speeds; do 
	
	speed=$(echo "$speed" | tr -d '"')

	# verify if the estimator exists
	if [ ! -f "models/model_"$data_type"_speed_$speed.pt" ]; then
		
		# train the estimator
		python -m utils.estimator.$data_type.train
	fi
done

