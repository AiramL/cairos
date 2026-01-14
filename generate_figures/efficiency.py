import matplotlib.pyplot as plt
import pandas as pd

from pickle import load
from numpy import mean, std
from os import listdir

# plot a figure showing the efficiency of different strategies
def efficiency_plot(file_path="results/server/flwr/training",
                    dataset="CIFAR-10",
                    strategies=["fedavg","cairos", "cairos_pb"],
                    alphas=["5.0"],
                    framework="torch",
                    models=["RESNET10"], 
                    execution=1,
                    n_selected=[10,20,30,40,50],
                    i_epochs=10,
                    scenario='equal',
                    PLOT=False, 
                    language="en"):
    
    plt.figure(figsize=(14, 10))
 
    if language == "en":
    
        plt.xlabel("Number of Selected Clients (#)", fontsize=16)
        plt.ylabel("Efficiency (%)", fontsize=16)

    elif language == "pt":
    
        plt.xlabel("Número de Clientes Selecionados (#)", fontsize=16)
        plt.ylabel("Eficiência (%)", fontsize=16)

    for strategy in strategies:

        for model in models:

            efficiency_list = []
    
            for n_select in n_selected:

                for alpha in alphas:
                    
                    data = pd.read_csv(f'{file_path}/{strategy}/{dataset}/{alpha}/{framework}/{execution}/{i_epochs}/{n_select}/{scenario}/{model}/aggregation.csv')
                    total = data.iloc[:,1].sum()
                    print(total, n_select, strategy)
                    efficiency = total / (n_select * 50)
                    efficiency_list.append(efficiency)

            plt.plot(n_selected, 
                     efficiency_list, 
                     label=f'i_epochs: {i_epochs}, model: {model}, strategy: {strategy}')

            
    plt.xticks(n_selected, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)


    plt.savefig(f"figures/efficiency_iepochs_{i_epochs}_n_selec_{n_select}_{language}.png",
                dpi=300,
                bbox_inches='tight')

    if PLOT:

        plt.show()

def accuracy_plot_varying_i_epochs(file_path="results/clients/flwr/classification",
                        dataset="CIFAR-10",
                        strategies=["fedavg"],
                        alphas=["5.0"],
                        framework="torch",
                        models=["MOBILENETV2"], 
                        n_clients=20,
                        execution=2,
                        cid=1,
                        n_select=1,
                        i_epoch=[1,2,4,8,16],
                        PLOT=False, 
                        language="en"):
    
    plt.figure(figsize=(14, 10))
 
    if language == "en":
    
        plt.xlabel("Epoch (#)", fontsize=16)
        plt.ylabel("Accuracy (%)", fontsize=16)

    elif language == "pt":
    
        plt.xlabel("Época (#)", fontsize=16)
        plt.ylabel("Acurácia (%)", fontsize=16)

    for model in models:
    
        for i_epochs in i_epoch:
        
            for strategy in strategies:

                for alpha in alphas:
                    
                    if strategy == "fedavg":
                    
                        data = pd.read_csv(f'{file_path}/{strategy}/{dataset}/{alpha}/{framework}/{execution}/{n_select}/{i_epochs}/{model}/{cid}')
                        epochs = range(1,len(data.iloc[:,1])+1)
                        
                        plt.plot(epochs, 
                                 data.iloc[:,1]*100, 
                                 label=f'i_epochs: {i_epochs}, model: {model}, nc = {n_select}')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)


    plt.savefig(f"figures/accuracy_fixed_iepochs_{i_epochs}_n_selec_{n_select}_{language}.png",
                dpi=300,
                bbox_inches='tight')

    if PLOT:

        plt.show()

def accuracy_plot_varying_distribution_and_sampling(file_path="results/clients/flwr/classification",
                                                   dataset="CIFAR-10",
                                                   strategies=["fedavg"],
                                                   alphas=["5.0"],
                                                   framework="torch",
                                                   models=["MOBILENETV2"], 
                                                   n_clients=20,
                                                   execution=3,
                                                   cid=1,
                                                   n_select=1,
                                                   i_epoch=[1,2,4,8,16],
                                                   distribution="uniform",
                                                   PLOT=False, 
                                                   language="en"):
    
    plt.figure(figsize=(14, 10))
 
    if language == "en":
    
        plt.xlabel("Epoch (#)", fontsize=16)
        plt.ylabel("Accuracy (%)", fontsize=16)

    elif language == "pt":
    
        plt.xlabel("Época (#)", fontsize=16)
        plt.ylabel("Acurácia (%)", fontsize=16)

    for model in models:
    
        for i_epochs in i_epoch:
        
            for strategy in strategies:

                for alpha in alphas:
                    
                    if strategy == "fedavg":
                    
                        data = pd.read_csv(f'{file_path}/{strategy}/{dataset}/{alpha}/{framework}/{execution}/{n_select}/{i_epochs}/{distribution}/{model}/{cid}')
                        epochs = range(1,len(data.iloc[:,1])+1)
                        
                        plt.plot(epochs, 
                                 data.iloc[:,1]*100, 
                                 label=f'i_epochs: {i_epochs}, model: {model}, nc = {n_select}, distribution: {distribution}')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)


    plt.savefig(f"figures/accuracy_fixed_iepochs_{i_epochs}_n_selec_{n_select}_distribution_{distribution}_{language}.png",
                dpi=300,
                bbox_inches='tight')

    if PLOT:

        plt.show()

def accuracy_plot_varying_distribution_and_iepochs(file_path="results/clients/flwr/classification",
                                                   dataset="CIFAR-10",
                                                   strategies=["fedavg"],
                                                   alphas=["5.0"],
                                                   framework="torch",
                                                   models=["MOBILENETV2"], 
                                                   n_clients=20,
                                                   execution=3,
                                                   cid=1,
                                                   n_selected=[1,2,4,8,16],
                                                   i_epochs=1,
                                                   distribution="uniform",
                                                   PLOT=False, 
                                                   language="en"):
    
    plt.figure(figsize=(14, 10))
 
    if language == "en":
    
        plt.xlabel("Epoch (#)", fontsize=16)
        plt.ylabel("Accuracy (%)", fontsize=16)

    elif language == "pt":
    
        plt.xlabel("Época (#)", fontsize=16)
        plt.ylabel("Acurácia (%)", fontsize=16)

    for model in models:
    
        for n_select in n_selected:
        
            for strategy in strategies:

                for alpha in alphas:
                    
                    if strategy == "fedavg":
                    
                        data = pd.read_csv(f'{file_path}/{strategy}/{dataset}/{alpha}/{framework}/{execution}/{n_select}/{i_epochs}/{distribution}/{model}/{cid}')
                        epochs = range(1,len(data.iloc[:,1])+1)
                        
                        plt.plot(epochs, 
                                 data.iloc[:,1]*100, 
                                 label=f'i_epochs: {i_epochs}, model: {model}, nc = {n_select}, distribution: {distribution}')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)


    plt.savefig(f"figures/accuracy_fixed_iepochs_{i_epochs}_n_selec_{n_select}_distribution_{distribution}_{language}.png",
                dpi=300,
                bbox_inches='tight')

    if PLOT:

        plt.show()



if __name__ == "__main__":
    
    efficiency_plot()