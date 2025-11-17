# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
#             INRIA
#
# University: Universidade Federal do Rio de Janeiro (UFRJ) - Brazil  
#             Ecole Polytechnique - France
#

import random
import keras
import logging
import numpy as np

import threading

from typing import List
from abc import ABC, abstractmethod

from flwr.common import (
    FitRes,
    Status, 
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.strategy.aggregate import aggregate

from utils.distillation import Distiller
from utils.distillation import MultiTeacherDistiller


class Server(ABC):

    def __init__(self,
                 n_clients,
                 k_clients,
                 clients,
                 e_epochs,
                 server_name,
                 timestamp,
                 model=None):
         
        # number of clients to select per epoch
        self.n_clients = n_clients
        
        # number of available clients
        self.k_clients = k_clients

        # clients objects/address
        self.clients = clients

        # total number of epochs
        self.e_epochs = e_epochs

        # global model
        self.global_model = model

        # aggregation algorithm
        self.server_name = server_name

        # path to save files
        self.log_path = "logs/server/prototype/"+str(timestamp)+"/"
        self.model_path = "models/server/prototype/"+str(timestamp)+"/"

        self.logger = self._create_logger()

        self.logger.debug(f"creating server object {self}")

    def _create_logger(self):

        logger = logging.getLogger(f'logger_{self.server_name}')
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            handler = logging.FileHandler(self.log_path+self.server_name+'_server.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @abstractmethod
    def train():
        pass
    
    # random client selection
    def select_clients(self):
        
        selected_clients = []

        N = random.sample(range(len(self.clients)),
                          self.n_clients)

        self.logger.debug(f"selected clients {N}")

        for cid in N:

            selected_clients.append(self.clients[cid])

        return selected_clients
    
    # compare the weights of two models
    def verify_weights(self,
                       model_1,
                       model_2):
        
        return all(np.array_equal(layers_1, layers_2) for layers_1, layers_2 in zip(model_1, model_2))

    
    # original federated average algorithm
    def average_models(self, 
                       models):


        results = [ FitRes(parameters=ndarrays_to_parameters(model),
                           num_examples=1/len(models),
                           metrics={},
                           status=Status(code=Code.OK, message="Success"))  
                           for _, model in models ]

        weights_results = [ (parameters_to_ndarrays(fit_res.parameters), 
                                                    fit_res.num_examples)
                            for fit_res in results ] 
        
        aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        return parameters_aggregated
    
    # run global epoch with the selected clients subset
    # this function can be executed in parallel
    def run_epoch(self,
                  N):
        
        results = []
        threads = []

        for client in N:

            self.logger.debug(f"creating thread for client {client}")
            threads.append(threading.Thread(target=client.fit))

        for thread in threads:
            
            thread.start()
            
        for thread in threads:

            thread.join()


        for client in N:

            results.append((client.mid, 
                            client.model.get_weights()))
            
        
        return results
    
    def update_client(self, 
                      global_weights,
                      client):
        

        client.model.set_weights(global_weights)

        client.evaluate()
            
        client.update_epoch()
        
    # compare neural networks
    def compare_model_architecture(self,
                                   global_shapes: List[np.ndarray],
                                   client_shapes: List[np.ndarray]) -> bool:
        
        """ Vererify if the client has the same architecture or not """
        global_shapes = [w.shape for w in global_shapes]
        client_shapes = [w.shape for w in client_shapes]

        return (client_shapes == global_shapes)



class FedAVGServer(Server):

    # initialize object 
    def __init__(self,
                 *args,
                 **kwargs):

        super().__init__(*args,
                         server_name="fedavg",
                         **kwargs)
        
        self.logger.debug(f'model {type(self.global_model)}')

    
    # global training 
    def train(self):

        # global training loop
        for n in range(self.e_epochs):
            
            # select clients to train
            N = self.select_clients()

            # execute clients local train 
            results = self.run_epoch(N)

            self.logger.debug(f"finished results of epoch {n}, starting processing")

            ''' process results '''

            # aggregate the models
            global_weights = parameters_to_ndarrays(self.average_models(results))
        
            # update local models
            self.run_update(global_weights)

            self.global_model.set_weights(global_weights)
            self.global_model.save(self.model_path+"global_model_epoch_"+str(n)+".keras")

            

            
    


class FHeMDaServer(Server):

    # initialize object 
    def __init__(self,
                 *args,
                 x,
                 y,
                 alpha,
                 temperature,
                 distill_epochs,
                 **kwargs):
        
        super().__init__(*args,
                         server_name="fhemda",
                         **kwargs)


        ''' knowledge distillation dataset '''

        # dataset features
        self.x  = x

        # dataset labels
        self.y  = y

        self.alpha = alpha
        self.temperature = temperature
        self.distill_epochs =  distill_epochs

        # all available models on the system
        self.system_models = {}
        
        # populate the dictionary of different models on the system
        self.verify_models()

    
    # identify the different models on the system
    def verify_models(self):

        for client in self.clients:

            if not self.system_models:

                self.system_models[0] = keras.models.clone_model(client.model)
                client.set_mid(0)

            else:

                mid =  0
        
                while(mid < len(self.system_models.values())):

                    if(self.compare_model_architecture(self.system_models[mid].get_weights(),
                                                       client.model.get_weights())):
                        client.set_mid(mid)
                        break
                    
                    mid += 1

                if(mid == len(self.system_models.values())):
                    self.system_models[mid] = client.model
                    client.set_mid(mid)
    


    # KD implementation
    def global_knowledge_distillation(self, 
                                      teacher, 
                                      student): 
        
        distiller = MultiTeacherDistiller(student=student,
                                          teacher=teacher)

        distiller.compile(optimizer=keras.optimizers.Adam(),
                          metrics=[keras.metrics.SparseCategoricalAccuracy()],
                          student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          distillation_loss_fn=keras.losses.KLDivergence(),
                          alpha=self.alpha,
                          temperature=self.temperature)

        distiller.fit(self.x, 
                      self.y, 
                      epochs=self.distill_epochs)

        return distiller.student
    
    # MTKD implementation
    def local_knowledge_distillation(self, 
                                     teacher, 
                                     student):
        
        distiller = Distiller(student=student,
                              teacher=teacher) 

        distiller.compile(optimizer=keras.optimizers.Adam(),
                          metrics=[keras.metrics.SparseCategoricalAccuracy()],
                          student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          distillation_loss_fn=keras.losses.KLDivergence(),
                          alpha=self.alpha,
                          temperature=self.temperature)

        distiller.fit(self.x, 
                      self.y, 
                      epochs=self.distill_epochs)

        return distiller.student    
      
    def update_client(self):

        for client in self.clients:
            
            client.model.set_weights(self.system_models[client.mid].get_weights())        

    def evaluate_clients(self):

        threads = []

        for client in self.clients:
            
            threads.append(threading.Thread(target=client.evaluate))

        for thread in threads:

            thread.start()
        
        for thread in threads:

            thread.join()

    # global training 
    def train(self):

        # global training loop
        for n in range(self.e_epochs):
            
            # select clients to train
            N = self.select_clients()

            # execute clients local train 
            results = self.run_epoch(N)

            ''' process results '''

            # get the unique mids
            unique_ids = {mid 
                          for mid, _ in 
                          results }

            # store epoch models
            epoch_models = { mid:[] 
                            for mid in 
                            unique_ids }

            # group all simliar models
            for result in results:

                epoch_models[result[0]].append((None,
                                                result[1]))

            # average models with same architecture, but different from the global
            for key in epoch_models.keys():
                
                if key != 0 and len(epoch_models[key]) > 1:

                    # update global model parameters 
                    self.logger.debug(f'executing average for models with the architecture {key}')
                    global_weights = parameters_to_ndarrays(self.average_models(epoch_models[key]))
                    self.system_models[key].set_weights(global_weights)


            self.logger.debug(f'Starting distillation phases')
            # define teachers models to the MTKD
            teachers = []

            for mid in range(1,len(self.system_models.keys())):

                teachers.append(self.system_models[mid])
                self.logger.debug(f'Appending new model {mid} on teachers list')

            # execute multi-teacher knowledge distillation
            self.logger.debug(f"executing multi-teacher knowledge distillation with a {type(teachers)} of {len(teachers)} teachers")
            self.global_knowledge_distillation(student=self.system_models[0],
                                               teacher=teachers)
            
            
            # aggregate the models with same architecture with the global model
            if len(epoch_models[0]):

                epoch_models[0].append((None, self.system_models[0].get_weights()))
                self.logger.debug(f"models architectures: {type(epoch_models[0])}")
                self.logger.debug(f"executing aggregation with a {type(epoch_models[0])} of {len(epoch_models[0])} models")
                global_weights = parameters_to_ndarrays(self.average_models(epoch_models[0]))
                self.system_models[0].set_weights(global_weights)


            # execute knowledge distillation
            for key in self.system_models:

                new_model = self.local_knowledge_distillation(student=self.system_models[key],
                                                              teacher=self.system_models[0])
                
                self.system_models[key].set_weights(new_model.get_weights())

            # update local models
            self.update_client()

            # test the new model
            self.evaluate_clients()                
