# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
#             INRIA
#
# University: Universidade Federal do Rio de Janeiro (UFRJ) - Brazil  
#             Ecole Polytechnique - France
#

import skimage
import torch

import numpy as np

from pickle import load
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def load_data_client(dataset_name='CIFAR-10',   # dataset used on the system
                     clientID=1,                # client identifier
                     numClients=1,              # number of clients
                     trPer=0.8,                 # percentage of test samples
                     distribution="dirichlet",  # manual or dirichlet
                     alpha=0.5):                # dirichlet distribution parameter

    if distribution == "manual":

        X = np.array([],
                     dtype=np.float32)
        
        Y = np.array([],
                     dtype=np.float32)

        for i in range(0,10):

            X_train_current = np.array([],dtype=np.float32)
            Y_train_current = np.array([],dtype=np.float32)
            X_test_current = np.array([],dtype=np.float32)
            Y_test_current = np.array([],dtype=np.float32)

            X_train_current = np.asarray(load(open('datasets/'+
                dataset_name+'/class'+str(i)+'Train','rb')), dtype=np.float32)
            
            X_test_current = np.asarray(load(open('datasets/'+
                dataset_name+'/class'+str(i)+'Test','rb')), dtype=np.float32)

            Y_train_current = np.asarray(load(open('datasets/'+
                dataset_name+'/class'+str(i)+'TrainLabel','rb')), dtype=np.float32)


            Y_test_current = np.asarray(load(open('datasets/'+
                dataset_name+'/class'+str(i)+'TestLabel','rb')),dtype=np.float32)

            begin_slice_train = int(len(Y_train_current)/numClients*(clientID-1))
            end_slice_train = int(len(Y_train_current)/numClients*clientID)
            begin_slice_test = int(len(Y_test_current)/numClients*(clientID-1))
            end_slice_test = int(len(Y_test_current)/numClients*clientID)

            X_train_current = X_train_current[begin_slice_train:end_slice_train]
            Y_train_current = Y_train_current[begin_slice_train:end_slice_train]
            X_test_current = X_test_current[begin_slice_test:end_slice_test]
            Y_test_current = Y_test_current[begin_slice_test:end_slice_test]
    
            if len(X) == 0:

                X = np.concatenate((X_train_current,X_test_current))
                Y = np.concatenate((Y_train_current,Y_test_current))

            else:

                X = np.concatenate((X,X_train_current,X_test_current))
                Y = np.concatenate((Y,Y_train_current,Y_test_current))
    
        # normalize the data
        X /= 255 

        # reshape MNIST and FMNIST
        if dataset_name == "MNIST" or dataset_name == "FMNIST":
            
            X = skimage.transform.resize(X, (len(X), 32, 32, 1))

        X, Y = shuffle(X, Y, random_state=47527)

        x_train, x_test, y_train, y_test = train_test_split(X, 
                                                            Y, 
                                                            test_size=trPer, 
                                                            random_state=42, 
                                                            stratify=Y)
        
        return x_train, x_test, y_train, y_test
    
    elif distribution == "dirichlet":

        data_path = f"datasets/{dataset_name}/distributions/nclients_{numClients}/alpha_{alpha}/cliente_{clientID}.pkl"

        with open(data_path ,"rb") as reader:
            
            data = load(reader)

        return data[0], data[1], data[2], data[3]


def load_data_server(dataset_name,          # dataset used on the system
                     numClients,            # number of clients
                     alpha=0.5):            # dirichlet distribution parameter

    x_train = np.array([])
    y_train = np.array([])
    x_test  = np.array([])
    y_test  = np.array([])

    for clientID in range(numClients):  

        data_path = f"datasets/{dataset_name}/distributions/nclients_{numClients}/alpha_{alpha}/cliente_{clientID}.pkl"

        with open(data_path ,"rb") as reader:
            
            data = load(reader)
        
        if len(x_train):
            
            np.append(x_train, data[0])
            np.append(x_test, data[1])
            np.append(y_train, data[2])
            np.append(y_test, data[3])
        
        else:
            
            x_train = data[0]
            x_test  = data[1]
            y_train = data[2]
            y_test  = data[3]
    
    return x_train, y_train, x_test, y_test

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, x, y):

        self.x = torch.from_numpy(x).float().permute(0, 3, 1, 2)
        self.y = torch.from_numpy(y).long()

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        return self.x[index], self.y[index]
