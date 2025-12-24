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

from .resnet import ResNet18

from .custom_models import (
        Net,
        resnet10
)

from .flisbee import FlisbeeNet

from timeit import default_timer as timer

from utils.torch.utils import allocate_cuda

def build_model(features_shape=None,
                labels_shape=10,
                client_id=0,
                model_name="RESNET18",
                lr=0.1):

    model = criterion = optimizer = device = scheduler = None

    device = allocate_cuda()
     
    if model_name == "RESNET18":

        model = ResNet18(num_classes=labels_shape)

        criterion = nn.CrossEntropyLoss()
    
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    elif model_name == "RESNET34":

        model = torchvision.models.resnet34(weights=None)

        model.fc = nn.Linear(model.fc.in_features, 
                             labels_shape) 

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
    elif model_name == "MOBILENETV2":

        model = torchvision.models.mobilenet_v2(weights=None)

        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 
                                        labels_shape)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
  

    elif model_name == "RESNET10":
        
        model = resnet10(num_classes=labels_shape)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        

    elif model_name == "CNN":

        model = Net(num_classes=labels_shape)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
    elif model_name == "FLISBEE":

        model = FlisbeeNet(num_classes=labels_shape)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    else:
        
        raise ValueError('Model not implemented')
        
    return model, criterion, optimizer, device, scheduler


def mem_usage(msg="",
              device=None,
              logger=None):

    logger.debug(f"[{msg}] Memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB | "
          f"Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

def train(model, 
          n_epochs, 
          optimizer, 
          criterion,
          scheduler,
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

                images, labels = data
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                loss = criterion(model(images), labels)
                
                loss.backward()

                optimizer.step()
    
                running_loss += loss.item()

            else:

                logger.debug(f'data batch size less than 2: {len(data[0])}')
    
    scheduler.step()

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

def train_eval(model, 
               n_epochs, 
               optimizer, 
               criterion,
               scheduler,
               device,
               trainloader,
               testloader,
               RESULT_PATH,
               exec_id,
               logger):
    
    # """Train the model on the training set."""
    best_acc = 0
    running_loss = 0.0
    
    for epoch in range(n_epochs):
        
        model.train()
        
        # logger.debug(f'starting local epoch {epoch} with a data size of {len(trainloader)}')

        for index, data in enumerate(trainloader):
            
            if len(data[0]) >= 2:

                images, labels = data

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                loss = criterion(model(images), labels)
                
                loss.backward()

                optimizer.step()

                scheduler.step()
                
                running_loss += loss.item()
        
        
        test_acc, loss = evaluate(model,
                                  device,
                                  criterion,
                                  testloader,
                                  logger)

        print(f'acc : {test_acc}, loss: {loss}, epoch: {epoch}')

        logger.debug(f'accuracy {test_acc}, loss {loss}')

        if test_acc > best_acc:
            
            print(f'new best: {test_acc}')
            best_acc = test_acc

            with open(f"{RESULT_PATH}/{exec_id}", "w") as writer:
                
                writer.writelines(f"{test_acc:.9f}\n")

    avg_trainloss = running_loss / len(trainloader)
    
    return avg_trainloss


def get_weights(model):

    return [ val.cpu().numpy() for _, val in model.state_dict().items() ]
