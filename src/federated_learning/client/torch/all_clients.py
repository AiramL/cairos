import os
import copy
import threading
import pandas as pd
import numpy as np
from time import sleep
from utils.torch.utils import get_args_client

# Get parameters
args = get_args_client()

# Set Parameters
                                                    # default parameters
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
exec_id = args.exec_id                              # 0
strategy = args.strategy                            # fedavg
scenario = args.scenario                            # all_in_one
original_training = args.original_training          # False
max_timeout =  args.max_timeout                     # False
estimation_per_batch = args.estimation_per_batch    # False

import torch
import flwr as fl

from architectures.torch.implementation import build_model

from utils.torch.load_federated_data import (
        load_data_client,
        CustomDataset)

from utils.loader import load_config
from .client import FLClient

from utils.torch.utils import create_logger_client

cfg = load_config('config/config.yaml')


message_length = 800 * 1024 * 1024

threads = []
parameters = {}


for index in range(num_clients):

    parameters[index] = {}
    parameters[index]['logger'] = create_logger_client(LOG_PATH+MODEL+'/', 
                                                       index)


    parameters[index]['logger'].debug(f"Execution path: {os.getcwd()}.")
    parameters[index]['logger'].debug(f"Training with model architecture {MODEL} and dataset {DATASET}.")

    parameters[index]['logger'].debug("Loading dataset")
    parameters[index]['x_train'], x_test, parameters[index]['y_train'], y_test = load_data_client(dataset_name=DATASET, 
                                                        clientID=index, 
                                                        numClients=num_clients, 
                                                        alpha=alpha,
                                                        trPer=ts,
                                                        distribution="dirichlet") 


    if DATASET == "SIGN":

        parameters[index]['x_train'] = np.transpose(x_train, (0, 3, 2, 1))
        parameters[index]['x_test'] = np.transpose(x_test, (0, 3, 2, 1))

    parameters[index]['train_d'] = CustomDataset(parameters[index]['x_train'], 
                                                 parameters[index]['y_train'])

    test_dataset = CustomDataset(x_test, 
                                 y_test)

    parameters[index]['trainloader'] = torch.utils.data.DataLoader(parameters[index]['train_d'], 
                                              batch_size=bs, 
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(test_dataset, 
                                             batch_size=bs, 
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True)

    # load through put dataframe
    df = pd.read_csv(f"data/processed/speed2/{exec_id}.csv") 
    parameters[index]['tdf'] = df[df['Node ID'] == index]

    parameters[index]['logger'].debug("Building model")

    labels = cfg['datasets'][DATASET]['classes']
    parameters[index]['model'], parameters[index]['criterion'], parameters[index]['optimizer'], parameters[index]['device'], parameters[index]['scheduler'] = build_model(features_shape=None,
                                                                 labels_shape=labels,
                                                                 model_name=MODEL,
                                                                 lr=0.1)

    parameters[index]['logger'].debug("Starting training")
    print(f'starting client {index}')
    threads.append(threading.Thread(target=fl.client.start_client,
                                    kwargs={'server_address':f'{SERVER_IP}:{SERVER_PORT}', 
                                            'client':FLClient(cid=index,
                                                              mid=0,
                                                              model=parameters[index]['model'],
                                                              i_epochs=i_epochs,
                                                              model_name=MODEL,
                                                              batch_size=bs,
                                                              dataset=DATASET,
                                                              strategy=strategy,
                                                              model_path=MODEL_PATH,
                                                              result_path=RESULT_PATH,
                                                              computation_time_path=COMP_PATH,
                                                              logger=parameters[index]['logger'],
                                                              optimizer=parameters[index]['optimizer'],
                                                              criterion=parameters[index]['criterion'],
                                                              scheduler=parameters[index]['scheduler'],
                                                              trainloader=parameters[index]['trainloader'],
                                                              testloader=testloader,
                                                              throughput=parameters[index]['tdf'],
                                                              max_timeout=max_timeout,
                                                              estimation_per_batch=estimation_per_batch,
                                                              original_training=original_training,
                                                              real_timer=False,
                                                              device=parameters[index]['device']).to_client(),
                                                              'grpc_max_message_length':message_length}))


for thread in threads:
    
    thread.start()

for thread in threads:

    thread.join()
