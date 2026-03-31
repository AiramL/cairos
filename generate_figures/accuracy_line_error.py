import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def accuracy_line_plot_grouped_with_std(file_path="results/clients/flwr/classification",
                                        dataset="CIFAR-10",
                                        strategies=["fedavg", "cairos_pb", "cairos_pe"],
                                        alphas=["5.0"],
                                        framework="torch",
                                        models=["RESNET10"],
                                        execution=10,
                                        n_rep=3,
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
                    
                    # --- ITERAÇÃO SOBRE CLIENTES ---
                    for rep in range(1,n_rep+1):

                        try:
                            # Ajuste o path conforme sua estrutura real
                            path = f'{file_path}/{strategy}/{dataset}/{alpha}/{framework}/experiment_{rep}/{execution}/{n_select}/{i_epochs}/equal/{model}/{cid}'
                            
                            if os.path.exists(path):
                                
                                data = pd.read_csv(path, header=None)
                                
                                acc_series = data.iloc[:, 1].values
                                all_client_data.append(acc_series)
                        
                        except (FileNotFoundError, IndexError, pd.errors.EmptyDataError):

                            continue

                    # --- PROCESSAMENTO E PLOTAGEM ---
                    if all_client_data:

                        min_len = min(len(x) for x in all_client_data)
                        
                        # Trunca os dados para o tamanho mínimo comum (caso algum cliente tenha rodado menos)
                        trimmed_data = [x[:min_len] for x in all_client_data]
                        
                        # Cria matriz (Linhas = Clientes, Colunas = Rodadas)
                        data_matrix = np.array(trimmed_data) * 100 # Converte para %
                        
                        # Calcula Média e Desvio Padrão por Rodada (axis=0)
                        mean_acc = np.mean(data_matrix, axis=0)
                        std_acc = np.std(data_matrix, axis=0)
                        
                        epochs = range(1, min_len + 1)

                        # Plota a LINHA DA MÉDIA
                        plt.plot(epochs,
                                 mean_acc,
                                 linewidth=3, # Linha um pouco mais fina para ver melhor
                                 label=labels[strategy],
                                 color=colors[strategy],
                                 linestyle=style[strategy])
                        
                        # Plota a ÁREA DE ERRO (Sombra)
                        plt.fill_between(epochs, 
                                         mean_acc - std_acc, # Limite inferior
                                         mean_acc + std_acc, # Limite superior
                                         color=colors[strategy], 
                                         alpha=0.2) # Transparência para ver sobreposição
                    else:
                        print(f"Nenhum dado encontrado para estratégia {strategy}")

        # Formatação do gráfico
        plt.xlabel(xlabel_text, fontsize=26)
        plt.ylabel(ylabel_text, fontsize=26)

        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=26)

        # Garante diretório
        os.makedirs("figures", exist_ok=True)
        
        filename = f"figures/accuracy_line_timeout_{execution}_n_selec_{n_select}_{language}_avg_std.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo: {filename}")

        if PLOT:
            plt.show()

if __name__ == "__main__":

    accuracy_line_plot_grouped_with_std(execution=10)
    accuracy_line_plot_grouped_with_std(execution=50)
