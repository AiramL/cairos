import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def accuracy_line_plot(file_path="results/clients/flwr/classification",
                       dataset="CIFAR-10",
                       strategies=["fedavg", "cairos_pb", "cairos_pe"],
                       alphas=["5.0"],
                       framework="torch",
                       models=["RESNET10"],
                       execution=10,  
                       cid=1,        
                       n_selected=[10],
                       i_epochs=10,
                       PLOT=False,
                       language="pt"):
    
    labels = {"fedavg":"FedAvg",
              "cairos_pb": "CAIROS PB",
              "cairos_pe": "CAIROS PE"}
    
    style  = {"fedavg":"-",
              "cairos_pb": "--",
              "cairos_pe": "-."}
    
    colors = {"fedavg":"r",
              "cairos_pb": "b",
              "cairos_pe": "k"}

    # Configurações de idioma
    if language == "en":
        
        xlabel_text = "Round (#)"
        ylabel_text = "Accuracy (%)"

    elif language == "pt":
        xlabel_text = "Rodada (#)"
        ylabel_text = "Acurácia (%)"

    # Gera um gráfico separado para cada configuração de 'n_selected'
    for n_select in n_selected:
        plt.figure(figsize=(14, 10))

        # Itera sobre as estratégias para compará-las no mesmo gráfico
        for strategy in strategies:
            for model in models:
                for alpha in alphas:
                    try:
                        # Caminho do arquivo (ajustar extensão se necessário, assumindo .csv)
                        path = f'{file_path}/{strategy}/{dataset}/{alpha}/{framework}/{execution}/{n_select}/{i_epochs}/equal/{model}/{cid}'
                        
                        # Tenta ler sem header se o arquivo tiver apenas dados crus, ou ajuste conforme seu CSV
                        # Assumindo aqui que col 0 = epoch ou index, col 1 = accuracy
                        data = pd.read_csv(path, header=None) 
                        
                        # Cria eixo X baseado no número de linhas
                        epochs = range(1, len(data) + 1)
                        accuracy_values = data.iloc[:, 1] * 100 # Converte para %
                        
                        plt.plot(epochs, 
                                 accuracy_values, 
                                 linewidth=5, 
                                 label=labels[strategy],
                                 color=colors[strategy],
                                 linestyle=style[strategy]) 

                    except FileNotFoundError:

                        print(f"Arquivo não encontrado: {path}")

        # Formatação do gráfico
        plt.xlabel(xlabel_text, fontsize=26)
        plt.ylabel(ylabel_text, fontsize=26)
        
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=26)

        filename = f"figures/accuracy_line_timeout_{execution}_n_selec_{n_select}_{language}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Gráfico de linhas salvo: {filename}")

        if PLOT:

            plt.show()


def accuracy_bar_plot_grouped(file_path="results/clients/flwr/classification",
                              dataset="CIFAR-10",
                              strategies=["fedavg", "cairos_pb", "cairos_pe"],
                              alphas=["5.0"],
                              framework="torch",
                              models=["RESNET10"],
                              execution=[5, 10, 20, 50],
                              cid=1,
                              n_selected=[10],
                              i_epochs=10,
                              PLOT=False,
                              language="pt"):
    
    labels = {"fedavg":"FedAvg",
              "cairos_pb": "CAIROS PB",
              "cairos_pe": "CAIROS PE"}
 
    colors = {"fedavg":"r",
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
        
        data_to_plot = {s: [] for s in strategies}

        # Coleta de dados
        for strategy in strategies:
            for timeout in execution:
                model = models[0]
                alpha = alphas[0]
                
                try:
                    path = f'{file_path}/{strategy}/{dataset}/{alpha}/{framework}/{timeout}/{n_select}/{i_epochs}/equal/{model}/{cid}'
                    data = pd.read_csv(path, header=None)
                    
                    final_acc = data.iloc[-1, 1] * 100 
                    data_to_plot[strategy].append(final_acc)
                    
                except (FileNotFoundError, IndexError):
        
                    print(f"Dados ausentes para: {strategy} no timeout {timeout}")
                    data_to_plot[strategy].append(0)

        # Configuração das Barras
        x = np.arange(len(execution)) 
        width = 0.25 
        multiplier = 0 

        for strategy, accuracies in data_to_plot.items():
            offset = width * multiplier
            rects = plt.bar(x + offset, accuracies, width, label=labels[strategy], color=colors[strategy])
            multiplier += 1

        # Ajustes Visuais
        plt.xlabel(xlabel_text, fontsize=26)
        plt.ylabel(ylabel_text, fontsize=26)
        
        # Centralizar ticks
        center_adjustment = (width * (len(strategies) - 1)) / 2
        plt.xticks(x + center_adjustment, execution, fontsize=26)
        plt.yticks(fontsize=26)
        plt.legend(fontsize=26, loc='lower right') 
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 100) # Fixar eixo Y de 0 a 100% para acurácia

        filename = f"figures/bar_accuracy_iepochs_{i_epochs}_n_selec_{n_select}_{language}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        if PLOT:
            plt.show()


if __name__ == "__main__":

    #accuracy_bar_plot_grouped()
    accuracy_line_plot(execution=10)
    accuracy_line_plot(execution=50)
