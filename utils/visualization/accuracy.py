from pickle import load
import matplotlib.pyplot as plt
from numpy import mean, std

def accuracy_error_plot(file_path="results/classification/processed/",dataset="WiSec", PLOT=False, language="pt"):
    
    plt.figure(figsize=(12, 8))
    
    servers = ["random",
               "m_fastest"]

    results = {}
    
    for server in servers:
        
        with open(file_path+server+"/"+dataset+"_mean_model","rb") as loader:
            result_list = load(loader)
            results[server+"mean"] = result_list*100
        
        with open(file_path+server+"/"+dataset+"_std_model","rb") as loader:
            result_list = load(loader)
            results[server+"std"] = result_list*100
    
    epochs = range(1,len(results[server+"mean"])+1)
     
    for server in servers:
        if server == "random":
            if language == "en":
                plt.errorbar(epochs, results[server+"mean"], yerr=results[server+"std"], capsize=3, label="Random and TOFL")

            elif language == "pt":
                plt.errorbar(epochs, results[server+"mean"], yerr=results[server+"std"], capsize=3, label="Aleatório e TOFL")

        else:
            plt.errorbar(epochs, results[server+"mean"], yerr=results[server+"std"], capsize=3, label="M-Fastest")
    
    
    if language == "en":

        plt.xlabel("Epoch (#)")
        plt.ylabel("Accuracy (%)")

    elif language == "pt":
        plt.xlabel("Época (#)")
        plt.ylabel("Acurácia (%)")
    
    plt.legend()
    plt.savefig("figures/"+dataset+"_accuracy.png",dpi=300,bbox_inches='tight')

    if PLOT:
        plt.show()

if __name__ == "__main__":
    
    datasets = ["VeReMi", "WiSec"]

    for dataset in datasets:
    
        accuracy_error_plot("results/classification/processed/", dataset)
        
