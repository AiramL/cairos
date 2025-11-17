#
# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#
import numpy as np
import os

import skimage
from pickle import load
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from torchvision import datasets, transforms


def load_data(dataset_name,
              clientID,
              numClients,
              trPer,
              distribution="manual",
              alpha=0.01):

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
    

def load_data_server(dataset_name="CIFAR-10",          # dataset used on the system
                     numClients=2,                     # number of clients
                     alpha=0.5):                       # dirichlet distribution parameter

    x_train = np.array([])
    y_train = np.array([])

    for clientID in range(numClients):  

        data_path = f"datasets/{dataset_name}/distributions/nclients_{numClients}/alpha_{alpha}/cliente_{clientID}.pkl"

        with open(data_path ,"rb") as reader:
            
            data = load(reader)
        
        if len(x_train):
            
            np.append(x_train, data[0])
            np.append(y_train, data[2])
        
        else:
            
            x_train = data[0]
            y_train = data[2]
    
    return x_train, y_train
