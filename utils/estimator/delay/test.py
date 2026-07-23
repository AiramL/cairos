import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import load_config
from .data import (
    load_tp,
    create_dataset
)

from .architecture import LSTM

def test(tpd, speed=0, payload_size=5.0):
    
    train_size = int(len(tpd) * 0.67)
    train, test = tpd[:train_size], tpd[train_size:]

    lookback = 5
    
    # Pass payload_size to calculate true delays
    X_train, y_train = create_dataset(train, lookback=lookback, payload_size=payload_size)
    X_test, y_test = create_dataset(test, lookback=lookback, payload_size=payload_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM()
    
    # Load the correct updated model file
    model.load_state_dict(torch.load(f"models/model_delay_speed_{speed}.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Get true labels for the full timeline to plot against
        true_delays = np.concatenate([y_train.numpy(), y_test.numpy()]).flatten()
        
        # Get predictions (since we fixed lstm.py, no slicing is needed!)
        train_pred = model(X_train.to(device)).cpu().numpy().flatten()
        test_pred = model(X_test.to(device)).cpu().numpy().flatten()
        
        # Prepare arrays padded with NaNs so the X-axis aligns correctly
        train_plot = np.ones_like(true_delays) * np.nan
        train_plot[:len(train_pred)] = train_pred
        
        test_plot = np.ones_like(true_delays) * np.nan
        test_plot[len(train_pred):] = test_pred
        
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(true_delays, c='b', label="Real Delay", alpha=0.5)
    plt.plot(train_plot, c='r', label="Training Prediction")
    plt.plot(test_plot, c='g', label="Testing Prediction")
    
    plt.xlabel("Time Window Sample (#)")
    plt.ylabel("Delay (Seconds)") # Changed from Throughput
    plt.title(f"Transmission Delay Test (Speed {speed}, Payload {payload_size}Mb)")
    plt.legend()
    plt.savefig(f"figures/test_delay_speed_{speed}_payload_{payload_size}.png")


if __name__ == "__main__":
   
    cfg = load_config('config/config.yaml') 
    speeds = cfg["simulation"]["speed"]["index"] 
    base_station_range = cfg["simulation"]["base_station"]["range"] # Copied from your train.py
    
    MODEL_PAYLOAD_SIZE = 0.5 

    for speed in speeds:
        # I added data_path back here so it loads identically to train.py
        tpu, tpd = load_tp(speed=speed, data_path=f"data/processed/{base_station_range}/speed")
        
        # Explicitly pass tpd and payload_size
        test(tpd=tpd, speed=speed, payload_size=MODEL_PAYLOAD_SIZE)