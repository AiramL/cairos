import torch
import time

import flwr as fl
from collections import OrderedDict
import numpy as np

from math import floor
from timeit import default_timer as timer
from collections import deque

from architectures.torch.implementation import (
        train, 
        evaluate)

from utils.estimator.throughput.architecture import EstimatorLSTM

# Federated Learning Client
class FLClient(fl.client.NumPyClient):

    def __init__(self, 
                 *args,
                 cid=-1,
                 model=None,
                 i_epochs=5,
                 model_name="MOBILENET",
                 batch_size=32,
                 dataset="CIFAR-10",
                 strategy="fedavg",
                 model_path="",
                 result_path="",
                 computation_time_path="",
                 logger=None,
                 optimizer=None,
                 criterion=None,
                 scheduler=None,
                 device=None,
                 trainloader=None,
                 testloader=None,
                 throughput=None,
                 max_timeout=120,
                 original_training=False,
                 estimation_per_batch=False,
                 real_timer=False,
                 deterministic_computational_delay=False,
                 computational_capacity=1000,
                 model_flops=200,
                 **kwargs):
        
        # paths
        self.original_training = original_training
        self.strategy = strategy
        self.dataset = dataset
        self.model_name = model_name
        self.model_path = model_path+model_name+'/'
        self.result_path = result_path+model_name+'/'
        self.time_path = computation_time_path+model_name+'/'
        self.logger = logger
        self.global_epoch = 0
        self.real_timer = real_timer
        self.computational_capacity_variance = 0.5

        # identifiers
        self.cid = cid
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.computational_capacity = computational_capacity
        self.model_flops = model_flops
        self.scheduler = scheduler
        self.deterministic_computational_delay = deterministic_computational_delay

        # client's data
        self.trainloader = trainloader 
        self.testloader = testloader 
        self.train_size = len(trainloader.dataset)
        self.test_size = len(testloader.dataset)

        # learning parameters
        self.i_epochs = i_epochs
        self.bs = batch_size

        # mobility parameters
        self.throughput = throughput

        # computing parameters for deterministic model
        if self.deterministic_computational_delay:
            
            self.batch_time = 0.1
            self.epoch_time = self.get_epoch_computational_delay()
    
        # communication parameters
        self.estimator = EstimatorLSTM()
        self.window_size = 5
        self.error_tolerance = 1.2
        self.message_period = 0.1
        self.estimation_frequency = 10 
        self.state = 0
        self.timeout = 0
        self.max_timeout = max_timeout
        self.estimation_per_batch = estimation_per_batch
        self.past_delays = deque(self.window_size*[0],                    # starting the past delays with a fixed value
                                 maxlen=self.window_size)                 # limiting window size
        self.estimated_past_delays = deque(self.window_size*[0],          # starting the estimated past delays with a fixed value
                                           maxlen=self.window_size)       # limiting window size
        self.model_size = sum(p.numel() * p.element_size()
                              for p in list(model.parameters()) + list(model.buffers())) / 1024

        self.logger.debug(f'starting client with id {self.cid}, for model {self.model_name}')
            
    def update_past_delays(self,
                           state):

        if state < self.window_size:

            begin = 0

        else:

            begin = state - self.window_size


        info = self.throughput.iloc[begin:state]

        for value in info['Throughput UL']:
                
            self.past_delays.appendleft(value)
    
    def time_to_state(self,
                      time):
        
        return int(time/self.message_period)

    # simulate the computation delay for a batch
    def get_batch_computational_delay(self):

        if self.deterministic_computational_delay:

            return self.batch_time
        
        else:
            
            return max(0.01, 
                       np.random.normal(self.model_flops / self.computational_capacity, 
                                        self.computational_capacity_variance))


    # simulate the computation delay for an epoch
    def get_epoch_computational_delay(self):

        if self.deterministic_computational_delay:

            return self.batch_time * len(self.trainloader)
        
        else:
            
            return max(0.01, 
                       np.random.normal(self.model_flops / self.computational_capacity, 
                                        self.computational_capacity_variance)) * len(self.trainloader)


    # use a gaussian generator to estimate the computation delay for a batch
    def get_batch_estimated_computational_delay(self):

        return max(0.01, 
                   np.random.normal(self.model_flops / self.computational_capacity, 
                                    self.computational_capacity_variance))


    # use a gaussian generator to estimate the computation delay for an epoch
    def get_epoch_estimated_computational_delay(self):
        
        return max(0.01, 
                    np.random.normal(self.model_flops / self.computational_capacity, 
                                     self.computational_capacity_variance)) * len(self.trainloader)



    def send_real_data_chunk(self, 
                             data,
                             state): 

        time_last_chunk = 0.0        
        
        self.logger.debug(f'sending real data of {data} bytes and at state {state}')
        throughput = self.throughput['Throughput UL'].iloc[state]

        maximum_chunk_size = floor(self.message_period * 
                                   1000 * 
                                   throughput)

        if (maximum_chunk_size >= data):

            time_last_chunk = data/(1000 * 
                                    throughput)
            
            return 0, time_last_chunk

        return data - maximum_chunk_size, time_last_chunk

    def send_estimated_data_chunk(self, 
                                  data): 

        time_last_chunk = 0.0

        window = torch.tensor(list(self.estimated_past_delays),
                              dtype=torch.float32).view(-1,1)
        
        estimated_delay = abs(self.estimator.predict(window))
        self.logger.debug(f'estimated delay: {estimated_delay}')
    
        self.logger.debug('updating estimated past delays')
        self.estimated_past_delays.appendleft(estimated_delay)

        maximum_chunk_size = floor(self.estimation_frequency * 
                                   1000 * 
                                   estimated_delay *
                                   self.message_period)

        self.logger.debug(f'verifying if there is remaning data, max chunck: {maximum_chunk_size}, data: {data}')
        if (maximum_chunk_size >= data):

            time_last_chunk = data/(1000 * 
                                    self.estimation_frequency *
                                    self.message_period *
                                    estimated_delay)
            
            return 0, time_last_chunk

        return data - maximum_chunk_size, time_last_chunk

    # simulate real communication delay
    def get_real_delay(self,
                       time):

        communication_time = 0
        remaining_data = self.model_size
        state = self.time_to_state(time)

        while (remaining_data):
            
            self.logger.implementation(f"remaining data: {remaining_data}")
            remaining_data, time_last_chunk = self.send_real_data_chunk(remaining_data,
                                                                        state)
            
            if remaining_data:
                
                state += 1
                communication_time += 1

        return float(self.message_period * (communication_time + time_last_chunk))

    
    # estimate the delay 
    def get_estimated_delay(self,
                            time):

        communication_time = 0
        remaining_data = self.model_size
        state = self.time_to_state(time)
        self.update_past_delays(state)
        self.estimated_past_delays = self.past_delays.copy()

        while (remaining_data):
            
            self.logger.debug(f"remaining data: {remaining_data}")
            remaining_data, time_last_chunk = self.send_estimated_data_chunk(remaining_data)
            self.logger.debug(f"remaining data after sending: {remaining_data}, time last chunk: {time_last_chunk}")
            
            if remaining_data:
                
                communication_time += 1

        return float(self.message_period * self.estimation_frequency * (communication_time + time_last_chunk))
    
    def update_global_epoch(self):

        self.logger.debug(f'updating epoch from {self.global_epoch} to {self.global_epoch+1}')
        self.global_epoch += 1

    def get_weights(self):
        
        result = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        return result
    
    def save_epoch_time(self,
                        epoch):

        with open(f"{self.time_path}training_time_{self.cid}.csv", "a") as writer:
            
            line = f'{epoch},{self.training_time}\n'

            writer.writelines(line)
        
    def set_weights(self, 
                    parameters):
    
        self.logger.debug(f'parameters id: {id(parameters)}')
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    def get_properties(self, 
                       config):
            
        return {'cid': self.cid}
   
    def train_cairos(self,
                     current_time):
        
        self.model.train()
        running_loss = 0.0
        stop = False
        
        for _ in range(self.i_epochs):
            
            if stop:

                break

            for data in self.trainloader:

                if len(data[0]) >= 2:

                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()

                    loss = self.criterion(self.model(images), labels)
                    
                    loss.backward()

                    self.optimizer.step()
        
                    running_loss += loss.item()

                else:

                    self.logger.debug(f'data batch size less than 2: {len(data[0])}')

                self.batch_time = self.get_batch_computational_delay()
                current_time += self.batch_time
                self.logger.debug(f'new current time: {current_time}')
                state = self.time_to_state(current_time)
                self.update_past_delays(state)

                # estimating communication delay for the next batch
                if self.estimation_per_batch:
                    
                    estimated_delay = self.get_batch_estimated_computational_delay()
                    current_time += estimated_delay
                    estimation_initial_time = time.time()
                    delay = self.get_estimated_delay(current_time)
                    self.logger.debug(f'estimation time: {time.time() - estimation_initial_time}')
                    self.logger.debug(f'estimated delay per batch: {delay}')
                    self.logger.debug(f'total estimated delay: {(delay + current_time) * self.error_tolerance }, timeout: {self.timeout}')
                    
                    if (delay + current_time) * self.error_tolerance > self.timeout:
                    
                        current_time -= estimated_delay
                        stop = True
                        break

                    else:
                    
                        current_time -= estimated_delay
            
            # estimating the communication delay for each local epoch
            if not self.estimation_per_batch:

                # estimating communication delay for the next epoch
                estimated_delay = self.get_epoch_estimated_computational_delay()
                current_time += estimated_delay
                delay = self.get_estimated_delay(current_time)
                self.logger.debug(f'estimated delay per epoch: {delay}')
                self.logger.debug(f'total estimated delay for next epoch: {delay * self.error_tolerance + current_time}, timeout: {self.timeout}')

                if (delay + current_time) * self.error_tolerance > self.timeout:
                    
                    current_time -= estimated_delay

                    break

                else:

                    current_time -= estimated_delay


        self.scheduler.step()
        
        avg_trainloss = running_loss / self.train_size
        
        return avg_trainloss, current_time, delay

    def fit(self, 
            parameters, 
            config):
        
        self.logger.debug('synchronizing clients')
        current_time = 0

        self.logger.debug('start timer to determine the computational time')
        if self.real_timer:

            fit_start = timer() + self.get_real_delay(current_time)

        else:
            
            # assuming we only start training when clients received the model
            fit_start = 0 
    
        self.logger.debug('calculating the timeout of this epoch')
        self.timeout = self.max_timeout - fit_start 
        self.logger.debug(f'timeout: {self.timeout}')

        # update weights 
        self.set_weights(parameters)

        # train model
        self.logger.debug("training model")
        if self.original_training:
            
            self.logger.debug("training with original FedAvg")
            loss = train(self.model, 
                         self.i_epochs, 
                         self.optimizer, 
                         self.criterion,
                         self.scheduler,
                         self.device,
                         self.trainloader,
                         self.logger)
            
            epoch_time = 0

            for _ in range(self.i_epochs):

                epoch_time += self.get_epoch_computational_delay()

            current_time = epoch_time + fit_start
            self.logger.debug(f'computation time: {current_time}, \
                                epoch time: {epoch_time}, \
                                fit start: {fit_start}')

        else:
            
            self.logger.debug("training with CAIROS")
            loss, current_time, estimated_delay = self.train_cairos(fit_start)
            self.logger.debug(f'total estimated delay after sending: {estimated_delay + current_time}')

        # Simulate that the client is transmitting the model through the network
        communication_time = self.get_real_delay(current_time)
        self.logger.debug(f'real communication delay: {communication_time}')

        # Determine client's computational time 
        if self.real_timer:
            
            self.training_time = timer() - fit_start + communication_time
            
            # Calculating the total local training time
            self.logger.debug(f'saving computational time')
            self.save_epoch_time(self.global_epoch)

        else:

            self.training_time = current_time + communication_time
        
        self.logger.debug(f'sending parameters to server: model_weights, \
                            len(train): {self.train_size} \
                            training time: {self.training_time}')
        
        return self.get_weights(), len(self.trainloader.dataset), {"time":self.training_time,'loss':loss,"cid":self.cid}

    def evaluate(self, 
                 parameters, 
                 config):
        
        self.logger.debug(f'evaluating model')  
        
        # update weights 
        self.set_weights(parameters)
 
        # evaluate model
        accuracy, loss = evaluate(self.model,
                                  self.device,
                                  self.criterion,
                                  self.testloader,
                                  self.logger)


        ''' Since all clients are selected to evaluate, we guaratee
            that each client knows the current global epoch number,
            to correctly read the delays' file input '''  
        self.update_global_epoch()

        # Calculating the total local training time
        self.logger.debug(f'saving results')
        with open(self.result_path+f"{self.cid}", "a") as writer:
                
                writer.writelines(str(self.global_epoch)+","+str(accuracy)+"\n")
        
        self.logger.debug(f'sending parameters to server: loss {loss}, len(test): {self.test_size} accuracy: {float(accuracy)}')

        return loss, self.test_size, {"accuracy": float(accuracy), "cid":self.cid}

