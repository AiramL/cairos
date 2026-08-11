
import os

import matplotlib.pyplot as plt

from utils.utils import load_config

from ..estimator.throughput.data import (
    load_tp,
)


def plot_throughput(speed=0,
                    base_station_range=1000, 
                    PLOT=True,
                    lang="pt"):

    if lang == "pt":

        plt.xlabel("Amostra (#)",fontsize=16)
        plt.ylabel("Vazão (Mb/s)",fontsize=16)
        

    elif lang == "en":
        
        plt.xlabel("Sample (#)",fontsize=16)
        plt.ylabel("Throughput (Mb/s)",fontsize=16)


    plt.plot(tpd,
             c='b')
        
    
    #plt.ylim(0,8)
    #plt.xlim(0,6000)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    figure_path = "figures/communication"

    os.makedirs(figure_path, 
                exist_ok=True)

    plt.savefig(f"{figure_path}/speed{speed}_bs_range_{base_station_range}_{lang}.png",
                dpi=300,
                bbox_inches='tight')
    
    if PLOT:
        
        plt.show()
    
    
if __name__ == "__main__":
   
    cfg = load_config('config/config.yaml') 

    speeds = cfg["simulation"]["speed"]["index"] 
    base_station_range = cfg["simulation"]["base_station"]["range"]

    for speed in speeds:

        tpu, tpd = load_tp(speed=speed,
                           data_path=f"data/processed/{base_station_range}/speed")
        
        plot_throughput(speed=speed,
                        base_station_range=base_station_range)
