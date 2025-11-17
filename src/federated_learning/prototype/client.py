# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
#             INRIA
#
# University: Universidade Federal do Rio de Janeiro (UFRJ) - Brazil  
#             Ecole Polytechnique - France
#


import logging


class Client():

    # initialize object
    def __init__(self,
                 cid,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 model,
                 b_batch,
                 i_epochs,
                 strategy,
                 timestamp):
        
        # identifiers
        self.cid = cid
        self.mid = -1 
        self.current_epoch = 1

        # model 
        self.model = model
        self.i_epochs = i_epochs
        self.b_batch = b_batch

        # client's data
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # server strategy
        self.strategy = strategy

        # path to save files
        self.log_path = "logs/clients/prototype/"+str(timestamp)+"/"
        self.model_path = "models/clients/prototype/"+str(timestamp)+"/"
        self.result_path = "results/classification/prototype/"+self.strategy+"/"+str(timestamp)+"/"

        self.logger = self._create_logger()

        self.logger.debug(f'creating client {self.cid}')

    def _create_logger(self):

        logger = logging.getLogger(f'logger_{self.cid}')
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            handler = logging.FileHandler(self.log_path+str(self.cid)+'.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger
    
    # set a new value for the mid
    def set_mid(self,
                mid):

        self.mid = mid
    
    def evaluate(self):

        with open(self.result_path+f'{self.cid}', "a") as writer:

            results = str(self.current_epoch)+","+ \
                      str(self.model.evaluate(self.x_test,
                                              self.y_test))+'\n'
            
            writer.writelines(results)

    def update_epoch(self):
        self.current_epoch += 1
        self.logger.debug(f'updating epoch: {self.current_epoch}')

    # local training 
    def fit(self):
        
        for _ in range(self.i_epochs):
            self.model.fit(self.x_train,
                           self.y_train,
                           batch_size=self.b_batch)

        self.logger.debug(f'saving model')
        self.model.save(self.model_path+str(self.cid)+"_epoch_"+str(self.current_epoch)+".keras")

        return (self.mid, 
                self.model.get_weights())

    

