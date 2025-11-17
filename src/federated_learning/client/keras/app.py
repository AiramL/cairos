
import os

import flwr as fl

from utils.keras.utils import get_args_client
from utils.keras.utils import create_logger_client
from utils.keras.load_federated_data import load_data
from architectures.keras.implementation import build_model

from utils.loader import load_config
from .client import FLClient

from utils.torch.utils import create_logger_client

cfg = load_config('configs/config.yaml')

system_models_ids = cfg['system_models_ids']

message_length = 800 * 1024 * 1024

# Get parameters
args = get_args_client()

# Set Parameters
                                                    # default parameters
client_id = args.client_id                          # 1
i_epochs = args.number_of_local_epochs              # 5
bs = args.batch_size                                # 32
ts = args.test_size                                 # 0.2
SERVER_IP = args.server_ip                          # [::]
SERVER_PORT = args.server_port                      # 8080
DATA_PATH = args.data_path                          # ../../datasets/VeReMi_Extension/mixalldata_clean.csv
DATASET_PATH = args.dataset_path                    # ../../datasets/VeReMi_Extension
DATASET = args.dataset                              # CIFAR-10         
MODEL_PATH = args.model_path                        # models/clients/flwr/
RESULT_PATH = args.result_path                      # results/clients/flwr/
LOG_PATH = args.log_path                            # logs/clients/flwr/
COMP_PATH = args.computation_time_path              # results/clients/flwr/computation_time
IMAGE_DATA = args.image_flag                        # 1
MODEL = args.model                                  # MOBILENET
num_clients = args.num_clients                      # 10
num_selected_clients = args.num_clients_fit         # 10
alpha = args.alpha                                  # 1
strategy = args.strategy                            # fedavg

logger = create_logger_client(LOG_PATH+MODEL+'/', 
                              client_id)

logger.debug(f"Execution path: {os.getcwd()}.")
logger.debug(f"Training with model architecture {MODEL} and dataset {DATASET}.")

logger.debug("Loading dataset")
x_train, x_test, y_train, y_test = load_data(dataset_name=DATASET, 
                                             clientID=client_id, 
                                             numClients=num_clients, 
                                             alpha=alpha,
                                             trPer=ts,
                                             distribution="dirichlet") 
spe = len(x_train)//bs


logger.debug("Building model")
features_shape = x_train.shape[1:]

if DATASET in ['CIFAR-10','MNIST','FMNIST']:

    labels = 10
    
if DATASET == 'CIFAR-100':

    labels = 100

else:

    pass

model = build_model(features_shape=features_shape,
                    labels_shape=labels,
                    model_name=MODEL,
                    loss='sparse_categorical_crossentropy',
                    lr=1e-3)

logger.debug("Starting training")
logger.debug(f'cid {client_id} with mid {system_models_ids[MODEL]} model {MODEL}')
fl.client.start_client(server_address=f'{SERVER_IP}:{SERVER_PORT}', 
                       client=FLClient(cid=client_id,
                                       mid=system_models_ids[MODEL],
                                       model=model,
                                       i_epochs=i_epochs,
                                       model_name=MODEL,
                                       batch_size=bs,
                                       dataset=DATASET,
                                       strategy=strategy,
                                       model_path=MODEL_PATH,
                                       result_path=RESULT_PATH,
                                       computation_time_path=COMP_PATH,
                                       x_train=x_train,
                                       y_train=y_train,
                                       x_test=x_test,
                                       y_test=y_test,
                                       logger=logger).to_client(),
                                       grpc_max_message_length=message_length)
