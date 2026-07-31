# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
#             INRIA
#
# University: Universidade Federal do Rio de Janeiro (UFRJ) - Brazil  
#             Ecole Polytechnique - France
#

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from .resnet import ResNet18

from .custom_models import (
        Net,
        resnet10
)
from utils.torch.utils import allocate_cuda

from .flisbee import FlisbeeNet

def build_model(features_shape:int=3,
                labels_shape:int=10,
                client_id:int=0,
                model_name:str="RESNET18",
                lr:float=0.1):

    model = criterion = optimizer = device = scheduler = None

    device = allocate_cuda()
     
    if model_name == "RESNET18":

        model = ResNet18(num_classes=labels_shape)
        
        if features_shape != 3:

            original = model.conv1

            model.conv1 = nn.Conv2d(features_shape,
                                    original.out_channels,
                                    kernel_size=original.kernel_size,
                                    stride=original.stride,
                                    padding=original.padding,
                                    bias=(original.bias is not None))

        criterion = nn.CrossEntropyLoss()
    
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    elif model_name == "RESNET34":

        model = torchvision.models.resnet34(weights=None)
        
        if features_shape != 3:

            original = model.conv1

            model.conv1 = nn.Conv2d(features_shape,
                                    original.out_channels,
                                    kernel_size=original.kernel_size,
                                    stride=original.stride,
                                    padding=original.padding,
                                    bias=(original.bias is not None))

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
        
        if features_shape != 3:

            original = model.features[0][0] #model.conv1

            model.features[0][0] = nn.Conv2d(features_shape,
                                    original.out_channels,
                                    kernel_size=original.kernel_size,
                                    stride=original.stride,
                                    padding=original.padding,
                                    bias=(original.bias is not None))

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
        
        if features_shape != 3:

            original = model.conv1

            model.conv1 = nn.Conv2d(features_shape,
                                    original.out_channels,
                                    kernel_size=original.kernel_size,
                                    stride=original.stride,
                                    padding=original.padding,
                                    bias=(original.bias is not None))

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        

    elif model_name == "CNN":

        model = Net(num_classes=labels_shape,
                    features_shape=features_shape)
        
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
    elif model_name == "FLISBEE":

        model = FlisbeeNet(num_classes=labels_shape,
                           in_channels=features_shape)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    elif model_name == "SHUFFLENET":

        model = torchvision.models.shufflenet_v2_x1_0(weights=None)

        if features_shape != 3:
        
            original = model.conv1[0]

            model.conv1[0] = nn.Conv2d(features_shape,
                                       24,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=(original.bias is not None))
        else:

            model.conv1[0] = nn.Conv2d(3, 
                                       24, 
                                       kernel_size=3, 
                                       stride=1, 
                                       padding=1, 
                                       bias=False)

        model.maxpool = nn.Identity()

        in_features_shuffle = model.fc.in_features
        
        model.fc = nn.Linear(in_features_shuffle, 
                             labels_shape)
        
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    elif model_name == "SQUEEZENET":

        model = torchvision.models.squeezenet1_1(weights=None)

        if features_shape != 3:
        
            original = model.features[0]

            model.features[0] = nn.Conv2d(features_shape,
                                    64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=(original.bias is not None))

        else:

            model.features[0] = nn.Conv2d(3,
                                        64,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        model.features[2] = nn.Identity()

        model.classifier[1] = nn.Conv2d(512, 
                                        labels_shape, 
                                        kernel_size=(1, 1), 
                                        stride=(1, 1))

        nn.init.normal_(model.classifier[1].weight, mean=0.0, std=0.01)
        nn.init.constant_(model.classifier[1].bias, 0.0)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    else:
        
        raise ValueError('Model not implemented')
        
    return model, criterion, optimizer, device, scheduler


def train(model, 
          n_epochs, 
          optimizer, 
          criterion,
          scheduler,
          device,
          trainloader,
          logger):
    
    model.train()
    running_loss = 0.0
    
    for epoch in range(n_epochs):
        
        logger.debug(f'starting local epoch {epoch} with a data size of {len(trainloader.dataset)}')

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
    
    avg_trainloss = running_loss / len(trainloader.dataset)
    
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

    inference_time_list = []

    with torch.no_grad():

        for data in testloader:
            
            initial_time = time.time()

            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            inference_time_list.append(time.time() - initial_time)
    
    return correct/total, loss, inference_time_list

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
               logger=None):
    
    # """Train the model on the training set."""
    best_acc = 0
    running_loss = 0.0

    batch_time_list = []
    epoch_time_list = []
    
    for epoch in range(n_epochs):
        
        model.train()
        epoch_start_time = time.time()

        for index, data in enumerate(trainloader):

            batch_start_time = time.time()
            
            if len(data[0]) >= 2:

                images, labels = data

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                loss = criterion(model(images), labels)
                
                loss.backward()

                optimizer.step()

                scheduler.step()
                
                running_loss += loss.item()

                batch_time_list.append(time.time() - batch_start_time)

        epoch_time_list.append(time.time() - epoch_start_time)

        test_acc, loss, inference_times = evaluate(model,
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
        
        writer.writelines(f"{best_acc:.9f}\n")

    
    with open(f"{RESULT_PATH}/batch_execution_time", "w") as writer:

        for batch_time in batch_time_list:
            
            writer.writelines(f"{batch_time:.9f}\n")

    with open(f"{RESULT_PATH}/epoch_execution_time", "w") as writer:

        for epoch_time in epoch_time_list:
            
            writer.writelines(f"{epoch_time:.9f}\n")

    with open(f"{RESULT_PATH}/inference_time", "w") as writer:

        for inference_time in inference_times:
            
            writer.writelines(f"{inference_time:.9f}\n")

    avg_trainloss = running_loss / len(trainloader.dataset)
    
    return avg_trainloss


def get_weights(model):

    return [ val.cpu().numpy() for _, val in model.state_dict().items() ]
