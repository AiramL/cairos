import yaml

def load_config(config_file:str="config/config.yaml"):

    with open(config_file,"r") as reader:

        return yaml.safe_load(reader)

