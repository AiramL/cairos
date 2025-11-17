import flwr as fl

from timeit import default_timer as timer

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
                 x_train=None,
                 y_train=None,
                 x_test=None,
                 y_test=None,
                 logger=None,
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

        self.model = model

        # client's data
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_size = len(self.x_train)
        self.test_size = len(self.x_test)

        # learning paramters
        self.i_epochs = i_epochs
        self.bs = batch_size
    
        self.logger.debug(f'starting client with id {self.cid}, mid {self.mid} for model {self.model_name}')

    def update_global_epoch(self):

        self.logger.debug(f'updating epoch from {self.global_epoch} to {self.global_epoch+1}')
        self.global_epoch += 1

    def get_parameters(self,
                       config):
        
        self.logger.debug('call of get_parameters successful')
        return self.model.get_weights()

    def save_epoch_time(self,
                        epoch):

        with open(f"{self.time_path}training_time_{self.cid}.csv", "a") as writer:
            
            line = f'{epoch},{self.training_time}\n'

            writer.writelines(line)

    def get_properties(self, 
                       config):
            
        self.logger.debug(f'call of get properties successful, returning mid: {self.mid}')
        return {'mid': self.mid}

    def fit(self, 
            parameters, 
            config):
        
        # Start timer to determine the computational time
        fit_start = timer()
        
        self.model.set_weights(parameters)

        self.logger.debug('training model')
        self.model.fit(self.x_train, 
                       self.y_train, 
                       epochs=self.i_epochs, 
                       batch_size=self.bs)

        self.logger.debug('saving the model')
        self.model.save(self.model_path+
                        str(self.cid)+
                        '_epoch_'+str(self.global_epoch)+
                        ".keras")
        
        # Determine client's computational time 
        self.training_time = timer() - fit_start
        

        # Calculating the total local training time
        self.logger.debug(f'saving computational time')
        self.save_epoch_time(self.global_epoch)
        
        self.logger.debug(f'sending parameters to server: model_weights, len(train): {self.train_size} mid: {self.mid}')
        return self.model.get_weights(), self.train_size, {"mid":self.mid}

    def evaluate(self, 
                 parameters, 
                 config):
        
        self.logger.debug(f'evaluating model')
        
        self.model.set_weights(parameters)
 
        loss, accuracy = self.model.evaluate(self.x_test, 
                                             self.y_test)

        ''' Since all clients are selected to evaluate, we guaratee
            that each client knows the current global epoch number,
            to correctly read the delays' file input ''' 
        
        self.update_global_epoch()

        # Calculating the total local training time
        self.logger.debug(f'saving results')
        with open(self.result_path+f"{self.cid}", "a") as writer:
                
                writer.writelines(str(self.global_epoch)+","+str(accuracy)+"\n")
        
        self.logger.debug(f'sending parameters to server: loss {loss}, len(test): {self.test_size} accuracy: {float(accuracy)} mid: {self.mid}')
        return loss, self.test_size, {"accuracy": float(accuracy),"mid":self.mid}

