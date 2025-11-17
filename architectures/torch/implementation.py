# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
#             INRIA
#
# University: Universidade Federal do Rio de Janeiro (UFRJ) - Brazil  
#             Ecole Polytechnique - France
#

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from .custom_models import (
        Net,
        resnet10
)

from timeit import default_timer as timer

def build_model(features_shape=(32,32,3),
                labels_shape=10,
                client_id=0,
                model_name="RESNET18",
                lr=1e-3):

    model = criterion = optimizer = device = None

    device = torch.device(f"cuda:{client_id%2}")
    
    if model_name == "RESNET18":

        model = torchvision.models.resnet18(weights=None)

        model.fc = nn.Linear(model.fc.in_features, 
                             labels_shape) 

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), 
                               lr=lr)
    
    elif model_name == "RESNET34":

        model = torchvision.models.resnet34(weights=None)

        model.fc = nn.Linear(model.fc.in_features, 
                             labels_shape) 

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), 
                               lr=lr)
        
    elif model_name == "MOBILENETV2":

        model = torchvision.models.mobilenet_v2(weights=None)

        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 
                                        labels_shape)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), 
                               lr=lr)
  

    elif model_name == "RESNET10":
        
        model = resnet10(num_classes=labels_shape)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), 
                               lr=lr)
        

    elif model_name == "CNN":

        model = Net(num_classes=labels_shape)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), 
                               lr=lr)
        

    else:
        
        pass
        
    return model, criterion, optimizer, device


def mem_usage(msg="",
              device=None,
              logger=None):

    logger.debug(f"[{msg}] Memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB | "
          f"Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

def train2(model, 
          n_epochs, 
          optimizer, 
          criterion, 
          device,
          trainloader,
          logger):
    
    """Train the model on the training set."""
    mem_usage("Before training", 
              device,
              logger)
    
    model.to(device)
    model.train()
    running_loss = 0.0
    
    for epoch in range(n_epochs):
        mem_usage(f"Epoch {epoch+1}/{n_epochs}", 
                  device,
                  logger)
        logger.debug(f'starting local epoch {epoch} with a data size of {len(trainloader)}')

        for index, data in enumerate(trainloader):
            
            if len(data[0]) >= 2:

                mem_usage(f"Batch {index+1}/{len(trainloader)}", 
                          device,
                          logger)
                
                logger.debug(f'data index: {index}')
                
                load_time = timer()
                images, labels = data
                logger.debug(f'load time: {timer() - load_time}')

                device_load_time = timer()
                images, labels = images.to(device), labels.to(device)
                logger.debug(f'device load time: {timer() - device_load_time}') 

                optimizer.zero_grad()

                mem_usage(f"After loading data", 
                          device,
                          logger)
                
                forward_time = timer()
                loss = criterion(model(images), labels)
                logger.debug(f'forward time: {timer() - forward_time}')
                
                backward_time = timer()
                loss.backward()
                logger.debug(f'backward time: {timer() - backward_time}')
                mem_usage(f"After backward pass", 
                          device,
                          logger)   

                step_time = timer()
                optimizer.step()
                logger.debug(f'step time: {timer() - step_time}')
                mem_usage(f"After optimizer", 
                          device,
                          logger)
    
                running_loss += loss.item()

            else:

                logger.debug(f'data batch size less than 2: {len(data[0])}')

    avg_trainloss = running_loss / len(trainloader)
    
    return avg_trainloss


def train(model, 
          n_epochs, 
          optimizer, 
          criterion, 
          device,
          trainloader,
          logger):
    
    """Train the model on the training set."""
    mem_usage("Before training", 
              device,
              logger)
    
    model.train()
    running_loss = 0.0
    
    for epoch in range(n_epochs):
        mem_usage(f"Epoch {epoch+1}/{n_epochs}", 
                  device,
                  logger)
        logger.debug(f'starting local epoch {epoch} with a data size of {len(trainloader)}')

        for index, data in enumerate(trainloader):
            
            if len(data[0]) >= 2:

                mem_usage(f"Batch {index+1}/{len(trainloader)}", 
                          device,
                          logger)
                
                logger.debug(f'data index: {index}')
                
                load_time = timer()
                images, labels = data
                logger.implementation(f'load time: {timer() - load_time}')

                device_load_time = timer()
                images, labels = images.to(device), labels.to(device)
                logger.implementation(f'device load time: {timer() - device_load_time}') 

                optimizer.zero_grad()

                mem_usage(f"After loading data", 
                          device,
                          logger)
                
                forward_time = timer()
                loss = criterion(model(images), labels)
                logger.implementation(f'forward time: {timer() - forward_time}')
                
                backward_time = timer()
                loss.backward()
                logger.implementation(f'backward time: {timer() - backward_time}')
                mem_usage(f"After backward pass", 
                          device,
                          logger)   

                step_time = timer()
                optimizer.step()
                logger.implementation(f'step time: {timer() - step_time}')
                mem_usage(f"After optimizer", 
                          device,
                          logger)
    
                running_loss += loss.item()

            else:

                logger.debug(f'data batch size less than 2: {len(data[0])}')

    avg_trainloss = running_loss / len(trainloader)
    
    return avg_trainloss


def evaluate(model,
             device,
             criterion,
             testloader,
             logger=None):

    model.eval()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        mem_usage(f"Before loading data", 
                    device,
                    logger)

        for data in testloader:
            
            mem_usage(f"During loading data", 
                    device,
                    logger)

            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    mem_usage(f"After loading data", 
              device,
              logger)

    return correct/total, loss


def get_weights(model):

    return [ val.cpu().numpy() for _, val in model.state_dict().items() ]
