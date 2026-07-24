import torch.nn as nn

class LSTM(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=1, 
                            hidden_size=50, 
                            num_layers=1, 
                            batch_first=True)

        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        # FIX: If input is unbatched (e.g., shape [5, 1]), add a batch dimension to make it [1, 5, 1]
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        lstm_out, _ = self.lstm(x)
        
        # Now lstm_out is guaranteed to be 3D: (batch_size, sequence_length, hidden_size)
        last_step_out = lstm_out[:, -1, :]
        
        predictions = self.linear(last_step_out)
        
        return predictions