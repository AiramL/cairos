# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
#             INRIA
#
# University: Universidade Federal do Rio de Janeiro (UFRJ) - Brazil  
#             Ecole Polytechnique - France
#

import keras

from os import makedirs
from datetime import datetime
from server import FHeMDaServer, FedAVGServer
from client import Client

from utils.load_federated_data import load_data, load_data_server

def main(server_name="fedavg",
         dataset_name="CIFAR-10",
         n_clients=5,
         k_clients=2,
         e_epochs=100,
         i_epochs=5,
         b_batch=32,
         distribution="dirichlet",
         alpha=0.5,
         trPer=0.8):

    clients = []

    tsp = datetime.now()
    timestamp = tsp.strftime('%Y-%m-%d-%H:%M:%S')

    # create paths
    server_log_path = "logs/server/prototype/"+str(timestamp)+"/"
    server_model_path = "models/server/prototype/"+str(timestamp)+"/"
    client_log_path = "logs/clients/prototype/"+str(timestamp)+"/"
    client_model_path = "models/clients/prototype/"+str(timestamp)+"/"
    results_path = "results/classification/prototype/"+server_name+"/"+str(timestamp)+"/"


    makedirs(server_log_path, 
             exist_ok=True)
        
    makedirs(server_model_path, 
             exist_ok=True)

    makedirs(client_log_path, 
             exist_ok=True)
        
    makedirs(client_model_path, 
             exist_ok=True)
    
    
    makedirs(results_path, 
             exist_ok=True)

    
    n_classes = 10
    input_shape = (32,32,3)

    model_1 = keras.applications.MobileNet(classes=n_classes, 
                                           input_shape=input_shape,
                                           weights=None,
                                           classifier_activation=None)
    
    model_2 = keras.applications.MobileNetV2(classes=n_classes, 
                                             input_shape=input_shape,
                                             weights=None,
                                             classifier_activation=None)
    
    model_1.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    
    
    model_2.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    
    

    system_models = {0:model_1,
                     1:model_2}


    if distribution == "manual":

            # use the same data for all clients
            x_train, x_test, y_train, y_test = load_data(dataset_name=dataset_name, 
                                                         clientID=1, 
                                                         numClients=1, 
                                                         trPer=trPer)

    if server_name == "fhemda":
        
        # initialize client
        for n in range(n_clients):
            
            model = keras.models.clone_model(system_models[n%2])
            model.set_weights(system_models[n%2].get_weights())
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            
            if distribution != "manual":

                x_train, x_test, y_train, y_test = load_data(dataset_name=dataset_name, 
                                                             clientID=n, 
                                                             numClients=n_clients, 
                                                             trPer=trPer,
                                                             distribution=distribution,
                                                             alpha=alpha)

            clients.append(Client(i_epochs=i_epochs,
                                  b_batch=b_batch,
                                  cid=n,
                                  model=model,
                                  x_train=x_train,
                                  y_train=y_train,
                                  x_test=x_test,
                                  y_test=y_test,
                                  strategy=server_name,
                                  timestamp=timestamp))

        # initialize server
        server = FHeMDaServer(n_clients=n_clients,
                              k_clients=k_clients,
                              clients=clients,
                              e_epochs=e_epochs,
                              x=x_train,  # this is equal to the last client distribution, verify how to define the dataset on the server side
                              y=y_train,
                              alpha=0.5,
                              temperature=10,
                              distill_epochs=5,
                              timestamp=timestamp)

    elif server_name == "fedavg":

        # initialize client
        for n in range(n_clients):

            model = keras.models.clone_model(model_1)
            model.set_weights(model_1.get_weights())
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            
            if distribution != "manual":

                x_train, x_test, y_train, y_test = load_data(dataset_name=dataset_name, 
                                                             clientID=n, 
                                                             numClients=n_clients, 
                                                             trPer=trPer,
                                                             distribution=distribution,
                                                             alpha=alpha)

            clients.append(Client(i_epochs=i_epochs,
                                  b_batch=b_batch,
                                  cid=n,
                                  model=model,
                                  x_train=x_train,
                                  y_train=y_train,
                                  x_test=x_test,
                                  y_test=y_test,
                                  strategy=server_name,
                                  timestamp=timestamp))
            
        x_train, y_train = load_data_server()

        # initialize server
        model = keras.models.clone_model(model_1)

        server = FedAVGServer(n_clients=n_clients,
                              k_clients=k_clients,
                              clients=clients,
                              e_epochs=e_epochs,
                              model=model,
                              timestamp=timestamp)



    # execute training
    server.train()

if __name__ == "__main__":

    # define server type
    for server_name in ["fhemda","fedavg"]:

        # execute the main function
        main(server_name)

