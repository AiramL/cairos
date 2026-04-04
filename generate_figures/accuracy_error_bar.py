import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.loader import load_config

def accuracy_bar_plot_grouped_with_std(file_path="results/clients/flwr/classification",
                                       dataset="CIFAR-10",
                                       strategies=["fedavg", "cairos_pb", "cairos_pe"],
                                       alphas=["5.0"],
                                       framework="torch",
                                       models=["RESNET10"],
                                       execution=[5, 10, 20, 50],
                                       n_clients=50,
                                       n_rep=3,
                                       cid=1,
                                       n_selected=[10],
                                       i_epochs=10,
                                       PLOT=False,
                                       language="pt"):

    labels = {"fedavg": "FedAvg",
              "cairos_pb": "CAIROS PB",
              "cairos_pe": "CAIROS PE"}

    colors = {"fedavg": "r",
              "cairos_pb": "b",
              "cairos_pe": "k"}

    if language == "en":
        xlabel_text = "Timeout (s)"
        ylabel_text = "Final Accuracy (%)"
    elif language == "pt":
        xlabel_text = "Tempo Máximo (s)"
        ylabel_text = "Acurácia (%)"

    for n_select in n_selected:
        plt.figure(figsize=(14, 10))

        means_to_plot = {s: [] for s in strategies}
        stds_to_plot = {s: [] for s in strategies}

        for strategy in strategies:
            for timeout in execution:
                model = models[0]
                alpha = alphas[0]
                
                client_accuracies = [] 

                for rep in range(n_rep):

                    try:
                        
                        path = f'{file_path}/{strategy}/{dataset}/{alpha}/{framework}/{timeout}/{i_epochs}/{n_select}/equal/{rep}/{model}/{cid}'
                        
                        if os.path.exists(path):
                            data = pd.read_csv(path, header=None)
                            final_acc = data.iloc[-1, 1] * 100
                            client_accuracies.append(final_acc)
                        
                        else:

                            pass

                    except (FileNotFoundError, IndexError, pd.errors.EmptyDataError):
                        continue

                if client_accuracies:

                    means_to_plot[strategy].append(np.mean(client_accuracies))
                    stds_to_plot[strategy].append(np.std(client_accuracies))

                else:

                    print(f"Dados ausentes para: {strategy} no timeout {timeout} (nenhum cliente encontrado)")
                    means_to_plot[strategy].append(0)
                    stds_to_plot[strategy].append(0)

        x = np.arange(len(execution))
        width = 0.25
        multiplier = 0

        for strategy in strategies:
            accuracies = means_to_plot[strategy]
            errors = stds_to_plot[strategy] 
            
            offset = width * multiplier
            
            rects = plt.bar(x + offset, accuracies, width, 
                            label=labels[strategy], 
                            color=colors[strategy],
                            yerr=errors,      
                            capsize=5)        
            multiplier += 1

        plt.xlabel(xlabel_text, fontsize=26)
        plt.ylabel(ylabel_text, fontsize=26)

        center_adjustment = (width * (len(strategies) - 1)) / 2
        plt.xticks(x + center_adjustment, execution, fontsize=26)
        plt.yticks(fontsize=26)
        plt.legend(fontsize=26, loc='upper left')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 100) 

        filename = f"figures/bar_accuracy_iepochs_{i_epochs}_n_selec_{n_select}_{language}_avg_std.png"
        
        os.makedirs("figures", exist_ok=True)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        if PLOT:
            plt.show()

if __name__ == "__main__":
    
    cfg = load_config()
    
    strategy = cfg['simulation']['federated_learning']['server']['strategy']
    dataset = cfg["simulation"]["federated_learning"]["client"]["dataset"]
    n_selected = cfg["simulation"]["federated_learning"]["server"]["n_clients_fit"]
    i_epochs = cfg["simulation"]["federated_learning"]["client"]["epochs"]
    rounds = cfg["simulation"]["federated_learning"]["server"]["rounds"]
    timeout = cfg["simulation"]["federated_learning"]["server"]["timeout"]
    n_rep = 1

    accuracy_bar_plot_grouped_with_std(strategies=[strategy],
                                       dataset=dataset,
                                       n_selected=[n_selected],
                                       i_epochs=i_epochs,
                                       n_rep=n_rep,
                                       execution=[timeout])
