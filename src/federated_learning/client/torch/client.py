import torch

import flwr as fl
from collections import OrderedDict


from timeit import default_timer as timer

from architectures.torch.implementation import train, evaluate

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, x, y):

        self.x = torch.from_numpy(x).float().permute(0, 3, 1, 2)
        self.y = torch.from_numpy(y).long()

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        return self.x[index], self.y[index]

# Federated Learning Client
class FLClient(fl.client.NumPyClient):

    def __init__(self, 
                 *args,
                 cid=-1,
                 mid=-1,
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
                 device=None,
                 trainloader=None,
                 testloader=None,
                 **kwargs):
        
        # paths
        self.strategy = strategy
        self.dataset = dataset
        self.model_name = model_name
        self.model_path = model_path+model_name+'/'
        self.result_path = result_path+model_name+'/'
        self.time_path = computation_time_path+model_name+'/'
        self.logger = logger
        self.global_epoch = 1
        
        # identifiers
        self.cid = cid
        self.mid = mid 

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # client's data
        self.trainloader = trainloader 
        self.testloader = testloader 
        self.train_size = len(trainloader.dataset)
        self.test_size = len(testloader.dataset)

        # learning parameters
        self.i_epochs = i_epochs
        self.bs = batch_size
    
        self.logger.debug(f'starting client with id {self.cid}, mid {self.mid} for model {self.model_name}')

    def update_global_epoch(self):

        self.logger.debug(f'updating epoch from {self.global_epoch} to {self.global_epoch+1}')
        self.global_epoch += 1

    def get_weights(self):
        
        self.logger.debug(f"GPU: {torch.cuda.current_device()}")
        result = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        return result


    def save_epoch_time(self,
                        epoch):

        with open(f"{self.time_path}training_time_{self.cid}.csv", "a") as writer:
            
            line = f'{epoch},{self.training_time}\n'

            writer.writelines(line)
        
    def set_weights(self, 
                    parameters):
    
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    def get_properties(self, 
                       config):
            
        self.logger.debug(f'call of get properties successful, returning mid: {self.mid}')
        
        return {'mid': self.mid,
                'cid': self.cid}

    def fit(self, 
            parameters, 
            config):
        
        # Start timer to determine the computational time
        fit_start = timer()
        
        # update weights 
        self.logger.debug(f"setting model parameters with {len(parameters)} layers")
        self.set_weights(parameters)

        # train model
        self.logger.debug("training model")
        self.logger.debug(f"GPU: {torch.cuda.current_device()}")
        loss = train(self.model, 
                     self.i_epochs, 
                     self.optimizer, 
                     self.criterion, 
                     self.device,
                     self.trainloader,
                     self.logger)
        
        self.logger.debug(f"GPU: {torch.cuda.current_device()}")

        # save model
        #self.logger.debug("Saving model")
        #torch.save(self.model.state_dict(), 
        #           self.model_path+
        #           str(self.cid)+
        #           '_epoch_'+str(self.global_epoch)+
        #           ".pth")
        #self.logger.debug(f"GPU: {torch.cuda.current_device()}")

        # making a simple test to the communication time
        communication_time = 2*self.cid
        # Determine client's computational time 
        self.training_time = timer() - fit_start + communication_time
        

        # Calculating the total local training time
        self.logger.debug(f'saving computational time')
        self.save_epoch_time(self.global_epoch)
        
        self.logger.debug(f'sending parameters to server: model_weights, len(train): {self.train_size} mid: {self.mid}')
        
        return self.get_weights(), len(self.trainloader.dataset), {"time":self.training_time, 'loss':loss, "cid":self.cid}

    def evaluate(self, 
                 parameters, 
                 config):
        
        self.logger.debug(f'evaluating model')  
        self.logger.debug(f"GPU: {torch.cuda.current_device()}")  
        
        # update weights 
        self.set_weights(parameters)
 
        # evaluate model
        accuracy, loss = evaluate(self.model,
                                  self.device,
                                  self.criterion,
                                  self.testloader,
                                  self.logger)
        
        self.logger.debug(f"GPU: {torch.cuda.current_device()}")


        ''' Since all clients are selected to evaluate, we guaratee
            that each client knows the current global epoch number,
            to correctly read the delays' file input ''' 
        
        self.update_global_epoch()

        # Calculating the total local training time
        self.logger.debug(f'saving results')
        with open(self.result_path+f"{self.cid}", "a") as writer:
                
                writer.writelines(str(self.global_epoch)+","+str(accuracy)+"\n")
        
        self.logger.debug(f'sending parameters to server: loss {loss}, len(test): {self.test_size} accuracy: {float(accuracy)} mid: {self.mid}')
        self.logger.debug(f"GPU: {torch.cuda.current_device()}")

        return loss, self.test_size, {"accuracy": float(accuracy), "mid":self.mid, "cid":self.cid}

