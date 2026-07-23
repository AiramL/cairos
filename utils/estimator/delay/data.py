import torch
import numpy as np
import pandas as pd

def create_dataset(dataset, lookback, payload_size=0.5, slot_duration=0.1):
    """Transform a time series into a delay prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps (N x 1)
        lookback: Size of window for prediction
        payload_size: Total size of the model/data to send to the cloud
        slot_duration: Time in seconds for each throughput sample (default 100ms)
    """
    X, y = [], []
    
    for i in range(len(dataset) - lookback):
        # 1. Feature is the historical window of throughputs
        feature = dataset[i : i + lookback]
        
        # 2. Future data to calculate the ground truth delay
        future_data = dataset[i + lookback : ]
        
        data_sent = 0.0
        time_elapsed = 0.0
        delay_t = -1.0 # Sentinel value to check if transmission finishes
        
        for th_array in future_data:
            # Your load_tp returns shape (N, 1), so we extract the scalar throughput
            th = th_array[0] 
            
            if th <= 0:
                time_elapsed += slot_duration
                continue
                
            data_in_slot = th * slot_duration
            
            if data_sent + data_in_slot >= payload_size:
                # Transmission finishes in this slot
                remaining_data = payload_size - data_sent
                fraction_of_slot = remaining_data / th
                delay_t = time_elapsed + fraction_of_slot
                break
            else:
                # Keep accumulating
                data_sent += data_in_slot
                time_elapsed += slot_duration
                
        # 3. Only keep the sample if the transmission actually finished before data ran out
        if delay_t != -1.0:
            X.append(feature)
            y.append([delay_t]) # Keep as a list so the tensor has shape (Samples, 1)
            
    # Convert directly to float32 tensors for PyTorch
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)


def load_tp(client_id=1, 
            data_path="data/processed/speed", 
            speed=0, 
            data_file="0.csv"):
    """Load throughput data for a specific client and speed.
    
    Args:
        client_id: The ID of the client to filter the data
        data_path: Path to the directory containing throughput CSV files
        speed: The speed category (used to select the correct subdirectory)
        data_file: The specific CSV file to load
    """
    
    
    df = pd.read_csv(f"{data_path}{speed}/{data_file}")
    dt = df[df['Node ID'] == client_id].reset_index()
    
    tpu = dt[['Throughput DL']].values.astype('float32')
    tpd = dt[['Throughput UL']].values.astype('float32')
    
    return tpu, tpd