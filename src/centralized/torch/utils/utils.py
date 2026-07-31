import argparse
import logging

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-ne","--number_of_epochs", type=int, default=100, help="How many local epochs")
    parser.add_argument("-b","--batch_size", type=int, default=128, help="Batch size to use during the federated learning training")
    parser.add_argument("-ts","--test_size", type=float, default=0.1, help="Test size to use") 
    parser.add_argument("-ds","--dataset", type=str, default="CIFAR-10", help="Dataset name")
    parser.add_argument("-mp","--model_path", type=str, default="models/centralized/", help="Path to the model")
    parser.add_argument("-rp","--result_path", type=str, default="results/centralized/", help="Path to store results")
    parser.add_argument("-lp","--log_path", type=str, default="logs/centralized/", help="Path to store logs")
    parser.add_argument("-md","--model", type=str, default="RESNET18", help="Model name to use in the FL scenario")
    parser.add_argument("-sct", "--save_comp_time", type=int, default=0, help="Flag to indicate if the compuitational time should be saved")

    return parser.parse_args()

def create_logger(log_path):
    
    logger = logging.getLogger(f'logger_centralized')
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:

        handler = logging.FileHandler(log_path+'centralized.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
