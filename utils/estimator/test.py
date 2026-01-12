import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.utils import load_config
from .data import (
        load_tp,
        create_dataset)

from .lstm import LSTM

def test(speed=0):
    
    train_size = int(len(tpd) * 0.67)
    test_size = len(tpd) - train_size
    train, test = tpd[:train_size], tpd[train_size:]

    lookback = 5
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM()
    model.load_state_dict(torch.load(f"models/model_10_speed_{speed}.pt", map_location=device))
    model.to(device)
    model.eval()
    
    with torch.no_grad():

        # shift train predictions for plotting
        train_plot = np.ones_like(tpd) * np.nan
        y_pred = model(X_train.to(device))
        y_pred = y_pred[:, -1, :]
        train_plot[lookback:train_size] = model(X_train.to(device))[:, -1, :].detach().cpu().numpy()
        # shift test predictions for plotting
        test_plot = np.ones_like(tpd) * np.nan
        test_plot[train_size+lookback:len(tpd)] = model(X_test.to(device))[:, -1, :].detach().cpu().numpy()
        
    plt.plot(tpd,c='b',label="real data")
    plt.plot(train_plot, c='r',label="traning")
    plt.plot(test_plot, c='g',label="testing")
    plt.xlabel("Sample (#)")
    plt.ylabel("Throughput (Mb/s)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
   
    cfg = load_config('config/config.yaml') 

    speeds = cfg["simulation"]["speed"]["index"] 

    for speed in speeds:

        tpu, tpd = load_tp(speed=speed)
        test(speed=speed)
