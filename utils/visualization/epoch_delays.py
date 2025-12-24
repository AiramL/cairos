from pickle import load
import matplotlib.pyplot as plt
from numpy import mean, std

from legends import legends_dicts

def selection_error_plot(n_clients=95, 
                         file_path="results/client_selection/processed/", 
                         PLOT=False, 
                         language="pt"):

    plt.figure(figsize=(14, 10))
    
    servers = ["random",
               "m_fastest",
               "tofl_oracle",
               "tofl_estimator_dl",
               "tofl_estimator_m_fastest"]
    
    legends = legends_dicts[language]

    if language == "en":
        
        plt.xlabel("Strategy")
        plt.ylabel("Global Epoch Delay (s)")

    elif language == "pt":

        plt.xlabel("Estratégia")
        plt.ylabel("Tempo de Trainamento de Época Global (s)")

    means = []
    stds = []


    for server in servers:
        with open(file_path+"server_"+server+"_n_clients_selected_"+str(n_clients)+"_mean","rb") as loader:
            result_list = load(loader)
            means.append(mean(result_list))
            stds.append(std(result_list))

    plt.bar([ legends[server] for server in servers ], 
            means, 
            yerr=stds, 
            capsize=3)
    
    plt.legend(fontsize=16)
    plt.savefig("figures/time_epoch"+str(n_clients)+"_"+language+".png",dpi=300,bbox_inches='tight')

if __name__ == "__main__":

    languages = ["pt", "en"]

    n_clients = [16, 95]

    for lang in languages:

        for n_client in n_clients:

            selection_error_plot(language=lang,
                                 n_clients=n_client)
