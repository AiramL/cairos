# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
#             INRIA
#
# University: Universidade Federal do Rio de Janeiro (UFRJ) - Brazil  
#             Ecole Polytechnique - France
#

import argparse
import logging
import pynvml
import torch 

def get_args_client():

    parser = argparse.ArgumentParser()

    parser.add_argument("-cid",
                        "--client_id", 
                        type=int, 
                        default=1,
                        help="Client identifier") 
    
    parser.add_argument("-eid",
                        "--exec_id", 
                        type=int, 
                        default=0,
                        help="Execution identifier") 
    
    parser.add_argument("-spid",
                        "--speed_id", 
                        type=int, 
                        default=0,
                        help="Speed identifier") 
    
    parser.add_argument("-bsr",
                        "--base_station_range", 
                        type=int, 
                        default=600,
                        help="Base station range in meters") 

    parser.add_argument("-nle",
                        "--number_of_local_epochs", 
                        type=int, 
                        default=5, 
                        help="How many updates the client does before to send the updated model to the server")
    
    parser.add_argument("-es",
                        "--scenario", 
                        type=str, 
                        default="all_in_one", 
                        help="Experimental scenario")
    
    parser.add_argument("-ot",
                        "--original_training", 
                        type=int, 
                        default=0, 
                        help="Experimental scenario")
    
    parser.add_argument("-epb",
                        "--estimation_per_batch", 
                        type=int, 
                        default=0, 
                        help="Estimate delay at each batch or not")
    
    parser.add_argument("-mt",
                        "--max_timeout", 
                        type=int, 
                        default=120, 
                        help="Maximun timeout at round begining")
    
    parser.add_argument("-rt",
                        "--real_timer", 
                        type=int, 
                        default=0, 
                        help="Use real hardware timer")


    parser.add_argument("-b",
                        "--batch_size", 
                        type=int, 
                        default=32, 
                        help="Batch size to use during the federated learning training")

    parser.add_argument("-ts",
                        "--test_size", 
                        type=float, 
                        default=0.2, 
                        help="Test size to use") 

    parser.add_argument("-sip",
                        "--server_ip", 
                        type=str, 
                        default="[::]", 
                        help="Server IP address")

    parser.add_argument("-sp",
                        "--server_port", 
                        type=str, 
                        default="8080", 
                        help="Server TCP port")

    parser.add_argument("-dp",
                        "--data_path", 
                        type=str, 
                        default="../../datasets/VeReMi_Extension/mixalldata_clean.csv", 
                        help="Path to the data")

    parser.add_argument("-dsp",
                        "--dataset_path", 
                        type=str, 
                        default="../../datasets/VeReMi_Extension", 
                        help="Dataset directory")

    parser.add_argument("-ds",
                        "--dataset", 
                        type=str, 
                        default="CIFAR-10", 
                        help="Dataset name")

    parser.add_argument("-mp",
                        "--model_path", 
                        type=str, 
                        default="models/clients/flwr/", 
                        help="Path to the model")

    parser.add_argument("-rp",
                        "--result_path", 
                        type=str, 
                        default="results/clients/flwr/", 
                        help="Path to store results")
    
    parser.add_argument("-ctp",
                        "--computation_time_path", 
                        type=str, 
                        default="results/clients/flwr/training/", 
                        help="Path to store results")

    parser.add_argument("-lp",
                        "--log_path", 
                        type=str, 
                        default="logs/clients/flwr/", 
                        help="Path to store logs")

    parser.add_argument("-imf",
                        "--image_flag", 
                        type=int, 
                        default=1, 
                        help="Indicates the type of data")

    parser.add_argument("-md",
                        "--model", 
                        type=str, 
                        default="MOBILENET", 
                        help="Model name to use in the FL scenario")

    parser.add_argument("-nc",
                        "--num_clients", 
                        type=int, 
                        default=10, 
                        help="How many clients are present during the training")

    parser.add_argument("-ncf",
                        "--num_clients_fit",
                         type=int, 
                         default=10, 
                         help="How many clients fits the model")
    
    parser.add_argument("-a",
                        "--alpha",
                         type=float, 
                         default=1, 
                         help="Alpha parameter of Dirichlet distribution")
 
    parser.add_argument("-str",
                        "--strategy", 
                        type=str, 
                        default="fedavg", 
                        help="Server aggregation strategy")

    return parser.parse_args()


IMPLEMENTATION_LEVEL = 2

def implementation(self, 
                   message, 
                   *args, 
                   **kwargs):
    
    if self.isEnabledFor(IMPLEMENTATION_LEVEL):
        self._log(IMPLEMENTATION_LEVEL, message, args, **kwargs)


def create_logger_client(log_path, 
                         cid):

    IMPLEMENTATION_LEVEL = 2
    logging.addLevelName(IMPLEMENTATION_LEVEL, "IMPLEMENTATION")
    logging.Logger.implementation = implementation


    logger = logging.getLogger(f'logger_{cid}')
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        
        handler = logging.FileHandler(log_path+str(cid)+'.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def allocate_cuda():

    if torch.cuda.is_available():
        
        n_cuda = torch.cuda.device_count()
        best = 'cuda'
        
        if n_cuda > 1:
           
            pynvml.nvmlInit()
            best_space = 0.
           
            for i in range(n_cuda):
               
                gpu = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(gpu)
                total = info.total
                free = info.free
                perc = free/total

                if perc > best_space:
                
                    best = f'cuda:{i}'
                    best_space = perc

        return torch.device(best)

    return torch.device('cpu')

def get_args_server():

    parser = argparse.ArgumentParser()

    parser.add_argument("-sip",
                        "--server_ip", 
                        type=str, 
                        default="[::]", 
                        help="Server IP address")

    parser.add_argument("-sp","--server_port", 
                        type=str, 
                        default="8080", 
                        help="Server TCP port")

    parser.add_argument("-sn",
                        "--server_name", 
                        type=str, 
                        default="fedavg", 
                        help="Server strategy")

    parser.add_argument("-nor",
                        "--number_of_rounds", 
                        type=int, 
                        default=3, 
                        help="How many global epochs the server executes")

    parser.add_argument("-nc",
                        "--num_clients", 
                        type=int, 
                        default=10, 
                        help="How many clients are present during the training")

    parser.add_argument("-ncf",
                        "--num_clients_fit", 
                        type=int, 
                        default=10, 
                        help="How many clients fits the model")

    parser.add_argument("-slp",
                        "--server_log_path", 
                        type=str, 
                        default="logs/server/flwr/", 
                        help="Path to save server logs")
    
    parser.add_argument("-a",
                        "--alpha",
                         type=float, 
                         default=1, 
                         help="Alpha parameter of Dirichlet distribution")
    
    parser.add_argument("-smp",
                        "--server_models_path", 
                        type=str, 
                        default="models/server/flwr/", 
                        help="Path to save server models")

    parser.add_argument("-ds",
                        "--dataset", 
                        type=str, 
                        default="CIFAR-10", 
                        help="Dataset name")

    parser.add_argument("-md",
                        "--model", 
                        type=str, 
                        default="MOBILENET", 
                        help="Model name to use in the FL scenario")

    parser.add_argument("-tp",
                        "--time_path", 
                        type=str, 
                        default="results/server/flwr/", 
                        help="Path to save the training time")

 
    parser.add_argument("-bk",
                        "--block_activation", 
                        type=str, 
                        default="uniform_aggregation", 
                        help="Control weights during aggregation and if do aggregation before the distillation phase")
    
    parser.add_argument("-to",
                        "--timeout", 
                        type=int, 
                        default=120,
                        help="Round timeout") 

    return parser.parse_args()

def create_logger_server(log_path):
    
    logger = logging.getLogger(f'logger_server')
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:

        handler = logging.FileHandler(log_path+'.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def limit_memory(logger):
    
    try:
        
        props = torch.cuda.get_device_properties(device=None)
        total_memory = props.total_memory
        client_memory = 1024 * 1024 * 1024 # 1024 MB for each client
        memory_percentage = client_memory/total_memory
        torch.cuda.set_per_process_memory_fraction(memory_percentage, 
                                                   device=None)
        
        logger.debug(f"Total memory on the system: {total_memory}.")
        logger.debug(f"Total memory percentage: {memory_percentage}.")
        logger.debug(f"Execution path: {os.getcwd()}.")
        logger.debug(f"Training with model architecture {MODEL} and dataset {DATASET}.")
        logger.debug(f"GPU: {torch.cuda.current_device()}")

    except:

        pass
