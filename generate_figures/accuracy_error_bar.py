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

        # Dicionários para armazenar médias e desvios padrão
        means_to_plot = {s: [] for s in strategies}
        stds_to_plot = {s: [] for s in strategies}

        for strategy in strategies:
            for timeout in execution:
                model = models[0]
                alpha = alphas[0]
                
                client_accuracies = [] 

                # Iterar sobre todos os clientes para calcular a média
                for rep in range(1,n_rep+1):
                    try:
                        # Nota: cid deve ser string se o nome da pasta for numérico
                        path = f'{file_path}/{strategy}/{dataset}/{alpha}/{framework}/experiment_{rep}/{timeout}/{n_select}/{i_epochs}/equal/{model}/{cid}'
                        
                        # Verifica se o arquivo existe antes de tentar ler (opcional, mas evita erros no pandas)
                        if os.path.exists(path):
                            data = pd.read_csv(path, header=None)
                            final_acc = data.iloc[-1, 1] * 100
                            client_accuracies.append(final_acc)
                        else:
                            # Se o arquivo não existe, decidimos não adicionar nada (ignorar cliente)
                            # ou você pode adicionar 0 se considerar falha.
                            pass

                    except (FileNotFoundError, IndexError, pd.errors.EmptyDataError):
                        continue

                # Calcular Média e Desvio Padrão para este timeout/estratégia
                if client_accuracies:
                    means_to_plot[strategy].append(np.mean(client_accuracies))
                    stds_to_plot[strategy].append(np.std(client_accuracies))
                else:
                    print(f"Dados ausentes para: {strategy} no timeout {timeout} (nenhum cliente encontrado)")
                    means_to_plot[strategy].append(0)
                    stds_to_plot[strategy].append(0)

        # --- Configuração das Barras ---
        x = np.arange(len(execution))
        width = 0.25
        multiplier = 0

        for strategy in strategies:
            accuracies = means_to_plot[strategy]
            errors = stds_to_plot[strategy] # Desvio padrão
            
            offset = width * multiplier
            
            # Adicionado yerr e capsize para o desvio padrão
            rects = plt.bar(x + offset, accuracies, width, 
                            label=labels[strategy], 
                            color=colors[strategy],
                            yerr=errors,      # Adiciona a barra de erro
                            capsize=5)        # Tamanho do "chapéu" da barra de erro
            multiplier += 1

        # --- Ajustes Visuais ---
        plt.xlabel(xlabel_text, fontsize=26)
        plt.ylabel(ylabel_text, fontsize=26)

        # Centralizar ticks
        center_adjustment = (width * (len(strategies) - 1)) / 2
        plt.xticks(x + center_adjustment, execution, fontsize=26)
        plt.yticks(fontsize=26)
        plt.legend(fontsize=26, loc='upper left')
        
        # Grid e Limites
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 100) 

        filename = f"figures/bar_accuracy_iepochs_{i_epochs}_n_selec_{n_select}_{language}_avg_std.png"
        
        # Garante que o diretório figures existe
        os.makedirs("figures", exist_ok=True)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        if PLOT:
            plt.show()

if __name__ == "__main__":
    
    cfg = load_config()
    
    strategy = cfg['simulation']['federated_learning']['server']['strategy']
    timeout = cfg['simulation']['federated_learning']['server']['timeout']

    accuracy_bar_plot_grouped_with_std(strategies=[strategy],
                                       execution=[timeout])
