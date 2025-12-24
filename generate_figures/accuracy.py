import matplotlib.pyplot as plt
import pandas as pd

from pickle import load
from numpy import mean, std
from os import listdir

# o que eu quero comparar?
# 1 - como varia a acurácia quando a quantidade de clientes participando varia?
# 2 - como varia a acurácia quando a quantidade de épocas locais varia?

def accuracy_plot_varying_selection(file_path="results/clients/flwr/classification",
                        dataset="CIFAR-10",
                        strategies=["fedavg"],
                        alphas=["5.0"],
                        framework="torch",
                        models=["MOBILENETV2"], 
                        n_clients=20,
                        execution=2,
                        cid=1,
                        n_selected=[1,2,4,8,16],
                        i_epochs=2,
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
    
    #for n in [1, 2, 4, 8, 16]:
    for n in [16]:

        accuracy_plot_varying_selection(i_epochs=n)
        #accuracy_plot_varying_i_epochs(n_select=n)
        #accuracy_plot_varying_distribution_and_iepochs(n_select=n)
        #accuracy_plot_varying_distribution_and_iepochs(i_epochs=n)
