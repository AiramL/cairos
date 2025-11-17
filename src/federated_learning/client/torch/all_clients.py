import gc
import os
import threading

import flwr as fl

from torch.utils.data import TensorDataset, DataLoader

from utils.utils import get_args_client
from utils.utils import create_logger
from utils.models import build_model
from utils.load_federated_data import load_data, CustomDataset
from utils.config import system_models_ids
from client import FLClient



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


message_length = 800 * 1024 * 1024

threads = []

for index in range(num_clients):

    logger = create_logger(LOG_PATH+MODEL+'/', 
                           index)


    logger.debug(f"Execution path: {os.getcwd()}.")
    logger.debug(f"Training with model architecture {MODEL} and dataset {DATASET}.")

    logger.debug("Loading dataset")
    x_train, x_test, y_train, y_test = load_data(dataset_name=DATASET, 
                                             clientID=index, 
                                             numClients=num_clients, 
                                             alpha=alpha,
                                             trPer=ts,
                                             distribution="dirichlet") 

    train_dataset = CustomDataset(x_train, 
                              y_train)

    test_dataset = CustomDataset(x_test, 
                             y_test)

    spe = len(x_train)//bs
    features_shape = x_train.shape[1:]

    del x_train, y_train, x_test, y_test
    gc.collect()


    trainloader = DataLoader(train_dataset, 
                            batch_size=bs, 
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False)

    testloader = DataLoader(test_dataset, 
                            batch_size=bs, 
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False)

    logger.debug("Building model")

    if DATASET in ['CIFAR-10',
                'MNIST',
                'FMNIST']:

        labels = 10
        
    if DATASET == 'CIFAR-100':

        labels = 100

    else:

        pass
    
    # need to create differnt models per client
    model, criterion, optimizer, device = build_model(features_shape=features_shape,
                                                    labels_shape=labels,
                                                    model_name=MODEL,
                                                    lr=1e-3)


    logger.debug("Starting training")
    print(f'starting client {index}')
    logger.debug(f'cid {index} with mid {system_models_ids[MODEL]} model {MODEL}')
    threads.append(threading.Thread(target=fl.client.start_client,
                                      kwargs={'server_address':f'{SERVER_IP}:{SERVER_PORT}', 
                                            'client':FLClient(cid=index,
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
                                            logger=logger,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            trainloader=trainloader,
                                            testloader=testloader,
                                            device=device).to_client(),
                                            'grpc_max_message_length':message_length}))


for thread in threads:
    
    thread.start()

for thread in threads:

    thread.join()
