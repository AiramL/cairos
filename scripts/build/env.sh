ENV_NAME=$(yq '.environment' "configs/config.yaml")

if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then

	echo "verify if conda $ENV_NAME environment exists"
	if conda env list | grep "$ENV_NAME " > /dev/null 2>&1; then 
		
		echo "$ENV_NAME already created"

	else

		conda create --name $ENV_NAME python=3.12
		conda activate $ENV_NAME
		python3.12 -m pip install -r requirements.txt

	fi	

fi
