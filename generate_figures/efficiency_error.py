import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.loader import load_config

def plot_efficiency_bar_with_error(file_path="results/server/flwr/training",
                                   dataset="CIFAR-10",
                                   strategies=["fedavg", "cairos_pe", "cairos_pb"],
                                   alphas=["5.0"],
                                   framework="torch",
                                   models=["RESNET10"],
                                   execution=[5, 10, 20, 50],
                                   n_selected=[10, 30],
                                   i_epochs=10,
                                   rounds=15,
                                   scenario='equal',
                                   n_rep=5,  
                                   PLOT=False,
                                   language="pt"):

    if language == "en":
        
        xlabel_text = "Timeout (s)"
        ylabel_text = "Efficiency (%)"

    elif language == "pt":
        
        xlabel_text = "Tempo Máximo por Rodada Global (s)"
        ylabel_text = "Eficiência (%)"

    labels = {"fedavg": "FedAvg",
              "cairos_pb": "CAIROS PB",
              "cairos_pe": "CAIROS PE"}

    colors = {"fedavg": "r",
              "cairos_pb": "b",
              "cairos_pe": "k"}

    for n_select in n_selected:
        plt.figure(figsize=(14, 10))

        means_to_plot = {s: [] for s in strategies}
        errors_to_plot = {s: [] for s in strategies}

        for strategy in strategies:
            
            for timeout in execution:
            
                model = models[0]
                alpha = alphas[0]
                
                rep_efficiencies = []

                for rep in range(n_rep):
            
                    try:
                        path = f'{file_path}/{strategy}/{dataset}/{alpha}/{framework}/{timeout}/{i_epochs}/{n_select}/{scenario}/{rep}/{model}/aggregation.csv'
                        
                        data = pd.read_csv(path,header=None)

                        total = data.iloc[:, 1].sum()
                        efficiency = total / (n_select * rounds) * 100

                        rep_efficiencies.append(efficiency)

                    except FileNotFoundError:

                        print(f"Arquivo não encontrado: {path}")
                        rep_efficiencies.append(0) 
                
                means_to_plot[strategy].append(np.mean(rep_efficiencies))
                errors_to_plot[strategy].append(np.std(rep_efficiencies))

        x = np.arange(len(execution))
        width = 0.25
        multiplier = 0

        for strategy in strategies:

            offset = width * multiplier
            
            means = means_to_plot[strategy]
            errors = errors_to_plot[strategy]

            plt.bar(x + offset, 
                    means, 
                    width, 
                    yerr=errors,      
                    capsize=5,        
                    label=labels[strategy], 
                    color=colors[strategy])
            
            multiplier += 1

        plt.xlabel(xlabel_text, fontsize=26)
        plt.ylabel(ylabel_text, fontsize=26)

        center_adjustment = (width * (len(strategies) - 1)) / 2
        plt.xticks(x + center_adjustment, execution, fontsize=26)
        plt.yticks(fontsize=26)

        plt.legend(fontsize=26, loc='best')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        filename = f"figures/bar_efficiency_avg_iepochs_{i_epochs}_n_selec_{n_select}_{language}.png"
        
        os.makedirs("figures", exist_ok=True)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        if PLOT:
            plt.show()
        else:
            plt.close() 

if __name__ == "__main__":
   
    cfg = load_config()

    dataset = cfg["simulation"]["federated_learning"]["client"]["dataset"]
    strategy = cfg["simulation"]["federated_learning"]["server"]["strategy"]
    n_selected = cfg["simulation"]["federated_learning"]["server"]["n_clients_fit"]
    i_epochs = cfg["simulation"]["federated_learning"]["client"]["epochs"]
    rounds = cfg["simulation"]["federated_learning"]["server"]["rounds"]
    timeout = cfg["simulation"]["federated_learning"]["server"]["timeout"]
    n_rep = 1

    plot_efficiency_bar_with_error(dataset=dataset,
                                   n_selected=[n_selected],
                                   i_epochs=i_epochs,
                                   strategies=[strategy],
                                   execution=[timeout],
                                   rounds=rounds,
                                   n_rep=n_rep)

