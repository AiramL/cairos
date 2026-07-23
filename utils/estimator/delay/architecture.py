from abc import ABC, abstractmethod
import torch
from .lstm import LSTM

class Estimator(ABC):
    def __init__(self, model):
        self.data = []
        self.model = model

    @abstractmethod
    def predict(self, data):
        pass

    def set_data(self, data):
        self.data = data

    def set_predictor(self, model):
        self.model = model


class EstimatorLSTM(Estimator):
    def __init__(self,
                 model_path="models",
                 speed=0):
        
        # Updated to load the new delay models
        model_name = f"{model_path}/model_delay_speed_{speed}.pt"
        
        model = LSTM()
        
        # Determine device dynamically just like in training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        
        model.load_state_dict(torch.load(model_name, map_location=self.device, weights_only=True))
        model.eval() # CRITICAL: Set model to evaluation mode for inference

        super().__init__(model)

    def predict(self, data):
        """
        Args:
            data: A list, numpy array, or tensor representing the lookback window.
        """
        with torch.no_grad():
            # Ensure data is a tensor and formatted as [batch=1, seq_len, features=1]
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            
            # Add batch and feature dimensions if they are missing
            if data.dim() == 1:
                data = data.view(1, -1, 1)
                
            data = data.to(self.device)
            
            # Predict
            pred = self.model(data)
            
            # Extract the single delay scalar safely
            if pred.dim() >= 2:
                pred = pred.view(-1)
                
            return float(pred[-1].cpu().item())