# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
#             INRIA
#
# University: Universidade Federal do Rio de Janeiro (UFRJ) - Brazil  
#             Ecole Polytechnique - France
#

import os

import torch
import numpy as np
import flwr as fl

from .strategy.fedavg import FedAvg
from utils.torch.utils import get_args_server
from utils.torch.utils import create_logger_server
from utils.torch.load_federated_data import load_data_server
from architectures.torch.implementation import build_model
from flwr.common import ndarrays_to_parameters

from architectures.torch.implementation import get_weights
from utils.torch.load_federated_data import CustomDataset
from utils.loader import load_config

cfg = load_config('config/config.yaml')

args = get_args_server()
                                                  # default parameters
num_rounds = args.number_of_rounds                # 3
server_ip = args.server_ip                        # '[::]:' 
server_port = args.server_port                    # '8080'
num_clients_fit = args.num_clients_fit            # 10
num_clients = args.num_clients                    # 10
aggregation = args.server_name                    # fedavg
server_log_path = args.server_log_path            # logs/server/flwr/
server_models_path = args.server_models_path      # models/server/flwr/
DATASET = args.dataset                            # CIFAR-10
alpha = args.alpha                                # 1.0
MODEL = args.model                                # MOBILENET             
time_path = args.time_path                        # results/server/flwr
block_activation = args.block_activation          # True
timeout = args.timeout                            # 120

os.makedirs(server_log_path, 
            exist_ok=True)

message_length = 800 * 1024 * 1024

props = torch.cuda.get_device_properties(device=None)
total_memory = props.total_memory
client_memory = 1024 * 1024 * 1024 # 1 GB for the server
memory_percentage = client_memory/total_memory
torch.cuda.set_per_process_memory_fraction(memory_percentage, 
                                           device=None)


logger = create_logger_server(log_path=server_log_path+aggregation)

logger.debug(f"Execution path: {os.getcwd()}.")

logger.debug(f"starting training with aggregation strategy {aggregation}, {num_clients} available clients, selecting {num_clients_fit} to fit, during {num_rounds} global epochs")

# Initialize model parameters
n_classes = cfg['datasets'][DATASET]['classes']

model, _, _, _, _ = build_model(labels_shape=n_classes,
                                model_name=MODEL)

ndarrays = get_weights(model)

parameters = ndarrays_to_parameters(ndarrays)

strategy = FedAvg(min_available_clients=num_clients,
                  min_fit_clients=num_clients_fit,
                  fraction_fit=0.1,
                  timeout=timeout,
                  logger=logger,
                  initial_parameters=parameters,
                  time_path=time_path)

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=num_rounds),
                       server_address=server_ip+":"+server_port,
                       strategy=strategy,
                       grpc_max_message_length=message_length)



