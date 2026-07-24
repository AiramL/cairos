speeds=$(yq '.simulation.speed.index[]' "config/config.yaml")

for speed in $speeds; do 
	
	speed=$(echo "$speed" | tr -d '"')

	# verify if the estimator exists
	if [ ! -f "models/model_10_speed_$speed.pt" ]; then
		
		# train the estimator
		python -m utils.estimator.delay.train
		python -m utils.estimator.throughput.train
	fi
done

