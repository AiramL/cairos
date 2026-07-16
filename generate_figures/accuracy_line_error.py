import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.loader import load_config

def accuracy_line_plot_grouped_with_std(file_path="results/clients/flwr/classification",
                                        dataset="CIFAR-10",
                                        strategies=["fedavg", "cairos_pb", "cairos_pe"],
                                        alphas=["5.0"],
                                        framework="torch",
                                        models=["RESNET10"],
                                        execution=10,
                                        n_rep=1,
                                        cid=1,
                                        n_selected=[10],
                                        i_epochs=10,
                                        PLOT=False,
                                        language="pt"):

    labels = {"fedavg": "FedAvg",
              "cairos_pb": "CAIROS PB",
              "cairos_pe": "CAIROS PE"}

    style = {"fedavg": "-",
             "cairos_pb": "--",
             "cairos_pe": "-."}

    colors = {"fedavg": "r",
              "cairos_pb": "b",
              "cairos_pe": "k"}

    # Configurações de idioma
    if language == "en":

        xlabel_text = "Round (#)"
        ylabel_text = "Accuracy (%)"

    elif language == "pt":

        xlabel_text = "Rodada (#)"
        ylabel_text = "Acurácia (%)"

    for n_select in n_selected:

        plt.figure(figsize=(14, 10))

        # Itera sobre as estratégias para compará-las no mesmo gráfico
        for strategy in strategies:

            for model in models:

                for alpha in alphas:
                    
                    all_client_data = [] 
                    
                    for rep in range(n_rep):

                        try:
                            
                            path = f'{file_path}/{strategy}/{dataset}/{alpha}/{framework}/{execution}/{n_select}/{i_epochs}/equal/{rep}/{model}/{cid}'
                            if os.path.exists(path):
                                
                                data = pd.read_csv(path, header=None)
                                
                                acc_series = data.iloc[:, 1].values
                                all_client_data.append(acc_series)
                        
                        except (FileNotFoundError, IndexError, pd.errors.EmptyDataError):

                            continue

                    if all_client_data:

                        min_len = min(len(x) for x in all_client_data)
                        
                        trimmed_data = [x[:min_len] for x in all_client_data]
                        data_matrix = np.array(trimmed_data) * 100 
                        mean_acc = np.mean(data_matrix, axis=0)
                        std_acc = np.std(data_matrix, axis=0)
                        
                        epochs = range(1, min_len + 1)

                        plt.plot(epochs,
                                 mean_acc,
                                 linewidth=3, 
                                 label=labels[strategy],
                                 color=colors[strategy],
                                 linestyle=style[strategy])
                        
                        plt.fill_between(epochs, 
                                         mean_acc - std_acc,
                                         mean_acc + std_acc, 
                                         color=colors[strategy], 
                                         alpha=0.2) 
                    else:

                        print(f"Nenhum dado encontrado para estratégia {strategy}")

        plt.xlabel(xlabel_text, fontsize=26)
        plt.ylabel(ylabel_text, fontsize=26)

        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=26)

        os.makedirs("figures", exist_ok=True)
        
        filename = f"figures/accuracy_line_timeout_{execution}_n_selec_{n_select}_{language}_avg_std.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        if PLOT:
            plt.show()

if __name__ == "__main__":
    
    cfg = load_config()

    accuracy_line_plot_grouped_with_std(execution=cfg['simulation']['federated_learning']['server']['timeout'],
                                        n_selected=[cfg['simulation']['federated_learning']['server']['n_clients_fit']],
                                        strategies=[cfg['simulation']['federated_learning']['server']['strategy']],
                                        i_epochs=cfg['simulation']['federated_learning']['client']['epochs'])
