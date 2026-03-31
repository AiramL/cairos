import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pickle import load
from numpy import mean, std
from os import listdir

def accuracy_plot_varying_selection(file_path="results/clients/flwr/classification",
                        dataset="CIFAR-10",
                        strategies=["fedavg"],
                        alphas=["5.0"],
                        framework="torch",
                        models=["RESNET10"], 
                        n_clients=50,
                        execution=2000,
                        cid=1,
                        dist="equal",
                        n_selected=[1,10,25,50],
                        i_epochs=5,
                        PLOT=False, 
                        language="pt"):
    
    plt.figure(figsize=(14, 10))
 
    if language == "en":
    
        plt.xlabel("Round (#)", fontsize=26)
        plt.ylabel("Accuracy (%)", fontsize=26)

    elif language == "pt":
    
        plt.xlabel("Rodada (#)", fontsize=26)
        plt.ylabel("Acurácia (%)", fontsize=26)

    labels = {1: "1 cliente selecionado",
              10: "10 clientes selecionados",
              25: "25 clientes selecionados",
              50: "50 clientes selecionados"}
    
    style  = {1:"-",
              10: "--",
              25: "-.",
              50: ":"}
    
    colors = {1:"r",
              10: "b",
              25: "k",
              50: "gray"}

    for model in models:
    
        for n_select in n_selected:
        
            for strategy in strategies:

                for alpha in alphas:
                    
                    if strategy == "fedavg":
                        
                        data = pd.read_csv(f'{file_path}/{strategy}/{dataset}/{alpha}/{framework}/{execution}/{i_epochs}/{n_select}/{dist}/{model}/{cid}')
                        epochs = range(1,len(data.iloc[:,1])+1)
                        
                        plt.plot(epochs, 
                                 data.iloc[:,1]*100, 
                                 linewidth=5, 
                                 color=colors[n_select],
                                 linestyle=style[n_select], 
                                 label=labels[n_select])

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26)


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
    
        plt.xlabel("Round (#)", fontsize=26)
        plt.ylabel("Accuracy (%)", fontsize=26)

    elif language == "pt":
    
        plt.xlabel("Rodada (#)", fontsize=26)
        plt.ylabel("Acurácia (%)", fontsize=26)

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

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26)


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
    
        plt.xlabel("Round (#)", fontsize=26)
        plt.ylabel("Accuracy (%)", fontsize=26)

    elif language == "pt":
    
        plt.xlabel("Rodada (#)", fontsize=26)
        plt.ylabel("Acurácia (%)", fontsize=26)

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

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26)


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
    
        plt.xlabel("Round (#)", fontsize=26)
        plt.ylabel("Accuracy (%)", fontsize=26)

    elif language == "pt":
    
        plt.xlabel("Rodada (#)", fontsize=26)
        plt.ylabel("Acurácia (%)", fontsize=26)

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

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26)


    plt.savefig(f"figures/accuracy_fixed_iepochs_{i_epochs}_n_selec_{n_select}_distribution_{distribution}_{language}.png",
                dpi=300,
                bbox_inches='tight')

    if PLOT:

        plt.show()

def accuracy_bar_plot_varying_selection(file_path="results/clients/flwr/classification",
                        dataset="CIFAR-10",
                        strategies=["fedavg"],
                        alphas=["5.0"],
                        framework="torch",
                        models=["RESNET10"], 
                        n_clients=50,
                        execution=2000,
                        cid=1,
                        dist="equal",
                        n_selected=[1,10,25,50],
                        i_epochs=5,
                        PLOT=False, 
                        language="pt"):
    
    plt.figure(figsize=(14, 10))
 
    if language == "en":
    
        plt.xlabel("Number of Selected Clients (#)", fontsize=26)
        plt.ylabel("Accuracy (%)", fontsize=26)

    elif language == "pt":
    
        plt.xlabel("Quantidade de Clientes Selecionados (#)", fontsize=26)
        plt.ylabel("Acurácia (%)", fontsize=26)

    labels = {1: "1 cliente selecionado",
              10: "10 clientes selecionados",
              25: "25 clientes selecionados",
              50: "50 clientes selecionados"}
    
    labels_l = ["1",
                "10",
                "25",
                "50"]
    
    style  = {1:"-",
              10: "--",
              25: "-.",
              50: ":"}
    
    colors = {1:"r",
              10: "b",
              25: "k",
              50: "gray"}

    colors_l = ["r",
                "b",
                "k",
                "gray"]
        
    results = []
    x = np.arange(len(n_selected)) 

    for model in models:
    
        for n_select in n_selected:
        
            for strategy in strategies:

                for alpha in alphas:
                    
                    if strategy == "fedavg":
                        
                        data = pd.read_csv(f'{file_path}/{strategy}/{dataset}/{alpha}/{framework}/{execution}/{i_epochs}/{n_select}/{dist}/{model}/{cid}', header=None)
                        
                        results.append(data.iloc[:,1][14]*100)
    
    plt.bar(x,
            results,
            linewidth=5,
            width=0.25,
            color=colors_l,
            tick_label=labels_l)
    
    plt.ylim(50, 75)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)


    plt.savefig(f"figures/accuracy_fixed_iepochs_{i_epochs}_n_selec_{n_select}_{language}.png",
                dpi=300,
                bbox_inches='tight')

    if PLOT:

        plt.show()



if __name__ == "__main__":
    
    #for n in [1, 2, 4, 8, 16]:
    #for n in [1, 10, 25, 50]:

        #accuracy_plot_varying_selection(i_epochs=n)
        #accuracy_plot_varying_i_epochs(n_select=n)
        #accuracy_plot_varying_distribution_and_iepochs(n_select=n)
        #accuracy_plot_varying_distribution_and_iepochs(i_epochs=n)
    #accuracy_plot_varying_selection()
    accuracy_bar_plot_varying_selection()
