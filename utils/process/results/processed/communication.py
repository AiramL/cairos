import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import (
        listdir,
        makedirs)

import sys 

from utils.utils import load_config

def plot_fig(mean, std):

    # Plot throughput
    plt.figure(figsize=(12, 8))
    
    plt.errorbar(executions, 
                 mean, 
                 yerr=std, 
                 capsize=3, 
                 fmt="r--o", 
                 ecolor = "black", 
                 label="throughputs_ul mean value")

    plt.xlabel("Number of Executions (#)")
    plt.ylabel("Throughput (Mb/s)")
    plt.legend()
    plt.show()


def generate_mean_and_std(n_executions=30, base_station_range=600, origin="mobility_0_"):

    dataset_name = f"data/raw/{base_station_range}/{origin}_simulation_"
    dataset_extension = ".csv"

    df = pd.read_csv(f"{dataset_name}{0}{dataset_extension}")

    for execution in range(n_executions):

        df = pd.concat((df, pd.read_csv(dataset_name+
                                        str(execution)+
                                        dataset_extension)))


    df_mean = df.groupby(df.index).mean()
    df_std = df.groupby(df.index).std()

    return (df_mean,df_std)



if __name__ == "__main__":

    cfg = load_config("config/config.yaml")
   
    speed = sys.argv[1]
    index = sys.argv[2]
    repetitions = cfg["simulation"]["communication"]["repetitions"]
    base_station_range = cfg['simulation']['base_station']['range']

    print("processing file ", index) 
    file_path = f"data/processed/{base_station_range}/speed{speed}/"
    file_name = f"{file_path}{index}.csv"
    makedirs(file_path, exist_ok=True)
    
    df_mean, df_std = generate_mean_and_std(repetitions,
                                            base_station_range,
                                            origin=f"mobility_{index}_speed_{speed}")

    df_mean.to_csv(file_name)

    print("processing finished")
