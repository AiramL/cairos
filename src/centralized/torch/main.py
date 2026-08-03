import os
import torch

import numpy as np

from .utils.utils import (
        get_args,
        create_logger)

from utils.torch.load_federated_data import (
        load_data_client,
        load_centralized_data,
        load_centralized_data_from_fl_dataset)

from torch.utils.data import Subset

from architectures.torch.implementation import (
        build_model,  
        evaluate,
        train_eval)

from utils.loader import load_config

def main(fraction:float=1.0,
         MODEL:str="RESNET18",
         exec_id:int=0,
         limited_data:bool=False,
         DATASET:str="CIFAR-10",
         from_fl:bool=True) -> None:

    # Get parameters
    args = get_args()
    rng = np.random.default_rng(seed=exec_id)

    # Set Parameters
                                                        # default parameters
    n_epochs = args.number_of_epochs                    # 100
    bs = args.batch_size                                # 128
    MODEL_PATH = args.model_path                        # models/centralized/
    RESULT_PATH = args.result_path                      # results/centralized/
    LOG_PATH = args.log_path                            # logs/centralized/


    LOG_PATH += f'{MODEL}/{limited_data}/{fraction}/{exec_id}/'

    os.makedirs(LOG_PATH, 
                exist_ok=True)
            
    os.makedirs(MODEL_PATH, 
                exist_ok=True)
    
    alpha = 5.0
   
    
    cfg = load_config('config/config.yaml')

    logger = create_logger(LOG_PATH)

    logger.debug(f"Training with model architecture {MODEL} and dataset {DATASET}.")

    logger.debug("Loading dataset")
    if limited_data:
        
        train, test = load_data_client(dataset_name=DATASET, 
                                       clientID=0, 
                                       numClients=40, 
                                       trPer=0.2,
                                       alpha=alpha,
                                       distribution="dirichlet") 

        scenario = "single_data_0"

    elif from_fl:
        
        scenario = "all_data_fl"

        train, test = load_centralized_data_from_fl_dataset(dataset_name=DATASET, 
                                                            alpha=alpha,
                                                            numClients=40)

    else:
    
        scenario = "all_data"

        train, test = load_centralized_data(dataset_name=DATASET)

    
    if fraction < 1.0:
        
        indexes = rng.choice(len(train),
                             int(fraction*len(train)),
                             replace=False)  

        train = Subset(train, indexes)
    
    RESULT_PATH += f'classification/{DATASET}/{alpha}/{scenario}/{fraction}/{MODEL}'

    os.makedirs(RESULT_PATH, 
                exist_ok=True)#for dataset in ["CIFAR-10"]:#, "SIGN", "CIFAR-100"]:

    logger.debug(f'trainset size {len(train)}, testset size {len(test)}')
    logger.debug(f'fraction: {fraction}')

    trainloader = torch.utils.data.DataLoader(train, 
                                              batch_size=bs, 
                                              shuffle=True,
                                              num_workers=2,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(test, 
                                             batch_size=100, 
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True)

    logger.debug("Building model")
    labels = cfg['datasets'][DATASET]['classes']
    features_shape = int(cfg['datasets'][DATASET]['features'][-1])
    print(f'features shape {features_shape}, labels shape {labels}')

    model, criterion, optimizer, device, scheduler = build_model(features_shape=features_shape,
                                                                 labels_shape=labels,
                                                                 model_name=MODEL,
                                                                 lr=0.1)    

    logger.debug("Training model") 
    model.to(device)
    running_loss = train_eval(model, 
                              100, 
                              optimizer, 
                              criterion,
                              scheduler,
                              device,
                              trainloader,
                              testloader,
                              RESULT_PATH,
                              exec_id,
                              logger)    

    logger.debug("Evaluating model")
    accuracy, _, _ = evaluate(model,
                              device,
                              criterion,
                              testloader,
                              logger)

    with open(f"{RESULT_PATH}/{exec_id}", "a") as writer:
                
        writer.writelines(f"{accuracy:.9f}\n")

if __name__ == "__main__":
   
    args = get_args()
    dataset = args.dataset

    for exec_id in range(1):

        for MODEL in ["RESNET10", "CNN", "RESNET18", "RESNET34", "MOBILENETV2", "FLISBEE", "SQUEEZENET", "SHUFFLENET"]:
        
            print(f'starting training with model {MODEL} at exec {exec_id} with dataset {dataset}')
            main(MODEL=MODEL,
                 exec_id=exec_id,
                 DATASET=dataset,
                 limited_data=True)

