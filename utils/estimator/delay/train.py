# implement the training for lstm model to predict delay
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data

from utils.utils import load_config
from .lstm import LSTM
from .data import (
    create_dataset,
    load_tp
)



def train(tpd, speed=0, payload_size=0.5, PLOT=False):
    # train-test split for time series
    train_size = int(len(tpd) * 0.67)
    train, test = tpd[:train_size], tpd[train_size:]

    lookback = 5
    # ADDED: payload_size is required for the new delay dataset
    X_train, y_train = create_dataset(train, lookback=lookback, payload_size=payload_size)
    X_test, y_test = create_dataset(test, lookback=lookback, payload_size=payload_size)

    print("Train shapes:", X_train.shape, y_train.shape)
    print("Test shapes:", X_test.shape, y_test.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM()
    model.to(device)

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), 
                             shuffle=True, 
                             batch_size=8)

    n_epochs = 20

    for epoch in range(n_epochs):
        model.train()

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
 
            y_pred = model(X_batch)
            
            # FIXED: Force shapes to match. 
            # If your LSTM outputs a sequence, grab only the final prediction
            if y_pred.dim() == 3: 
                y_pred = y_pred[:, -1, :] 
            
            # Ensure both are exactly [batch_size, 1]
            y_pred = y_pred.view(-1, 1)
            y_batch = y_batch.view(-1, 1)
            
            loss = loss_fn(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        if epoch % 5 == 0 or epoch == n_epochs - 1: # Print more often, and always on last epoch
            model.eval()
            with torch.no_grad():
                # Train RMSE
                y_train_pred = model(X_train.to(device)).view(-1, 1)
                train_rmse = torch.sqrt(loss_fn(y_train_pred, y_train.to(device).view(-1, 1)))
                
                # Test RMSE
                y_test_pred = model(X_test.to(device)).view(-1, 1)
                test_rmse = torch.sqrt(loss_fn(y_test_pred, y_test.to(device).view(-1, 1)))
            
            print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))


    # FIXED: Plotting logic rewritten for Delay instead of Throughput
    if PLOT:
        model.eval()
        with torch.no_grad():
            plt.figure(figsize=(12, 6))
            
            # Combine true labels for the full timeline
            true_delays = np.concatenate([y_train.numpy(), y_test.numpy()]).flatten()
            plt.plot(true_delays, c='b', label="Real Delay", alpha=0.5)
            
            # Prepare train plot
            train_pred = model(X_train.to(device)).view(-1).cpu().numpy()
            train_plot = np.ones_like(true_delays) * np.nan
            train_plot[:len(train_pred)] = train_pred
            
            # Prepare test plot
            test_pred = model(X_test.to(device)).view(-1).cpu().numpy()
            test_plot = np.ones_like(true_delays) * np.nan
            test_plot[len(train_pred):] = test_pred
            
            plt.plot(train_plot, c='r', label="Training Prediction")
            plt.plot(test_plot, c='g', label="Testing Prediction")
            
            plt.title(f"Transmission Delay Prediction (Payload: {payload_size} Mb)")
            plt.xlabel("Time Window Sample (#)")
            plt.ylabel("Delay (Seconds)")
            plt.legend()
            plt.show()
    
    torch.save(model.state_dict(),f"models/model_delay_speed_{speed}.pt")


if __name__ == "__main__":
    cfg = load_config('config/config.yaml') 
    speeds = cfg["simulation"]["speed"]["index"] 
    base_station_range = cfg["simulation"]["base_station"]["range"] 
    
    # Define your model size here
    MODEL_PAYLOAD_SIZE = 0.5 

    for speed in speeds:
        tpu, tpd = load_tp(speed=speed,
                           data_path=f"data/processed/{base_station_range}/speed")
        
        # Pass tpd and the payload size into the train function
        train(tpd=tpd, speed=speed, payload_size=MODEL_PAYLOAD_SIZE, PLOT=False)
