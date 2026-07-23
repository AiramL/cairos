import torch.nn as nn

class LSTM(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=1, 
                            hidden_size=50, 
                            num_layers=1, 
                            batch_first=True)

        # Output exactly 1 value (the transmission delay)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        # x input shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # SLICE HERE: Grab the hidden state of the very last timestep
        # lstm_out shape becomes: (batch_size, 50)
        last_step_out = lstm_out[:, -1, :]
        
        # Pass only the last timestep through the linear layer
        # Output shape: (batch_size, 1)
        predictions = self.linear(last_step_out)
        
        return predictions