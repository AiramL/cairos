from pickle import load
import matplotlib.pyplot as plt
from itertools import accumulate

from utils.utils import load_config 
from .legends import legends_dicts

def process_accuracy_delays(n_clients=95,
                            dataset="WiSec",
                            acc_path="results/classification/processed/",
                            time_path="results/client_selection/processed/",
                            n_executions=10,
                            model_size=500,
                            language="en",
                            servers=[]):
    
    plt.figure(figsize=(12, 8))
     
    legends = legends_dicts[language]
    
    if language == "en":
        
        plt.xlabel("Time (s)")
        plt.ylabel("Accuracy (%)")

    elif language == "pt":
        
        plt.xlabel("Tempo (s)")
        plt.ylabel("Acurácia (%)")


    results_time = { }
    results = { }
    
    for server in servers:
        
        with open(time_path+"server_"+server+"_n_clients_selected_"+str(n_clients)+"_mean","rb") as loader:
            
            result_list = load(loader)
            results_time[server] =  list(accumulate(result_list))

    for server in servers:
            
        if server == "m_fastest" or server == "tofl_estimator_m_fastest":
            
            mean_file = acc_path+"m_fastest/"+dataset+"_mean_model"
            std_file = acc_path+"m_fastest/"+dataset+"_std_model"
            
        else:
            
            mean_file = acc_path+"random/"+dataset+"_mean_model"
            std_file = acc_path+"random/"+dataset+"_std_model"
        
        with open(mean_file,"rb") as loader:
            
            result_list = load(loader)
            results[server+"mean"] = result_list*100
        
        with open(std_file,"rb") as loader:
        
            result_list = load(loader)
            results[server+"std"] = result_list*100
    
    for server in servers:
        
        if server == "m_fastest" or server == "tofl_estimator_m_fastest":
            
            ''' for these strategies, the results show that the convergence takes 
                6 epochs '''
            plt.errorbar(results_time[server][:6], 
                         results[server+"mean"][:6], 
                         yerr=results[server+"std"][:6], 
                         capsize=3, 
                         label=legends[server])

        else:
            
            ''' for these strategies, the results show that the convergence takes
                5 epochs '''
            plt.errorbar(results_time[server][:6], 
                         results[server+"mean"][:6], 
                         yerr=results[server+"std"][:6], 
                         capsize=3, 
                         label=legends[server])

    
    plt.legend()
    plt.savefig(f"figures/{dataset}_time2acc_n_clients_{n_clients}_{language}.png",dpi=300,bbox_inches='tight')

if __name__ == "__main__":

    
    cfg = load_config('config/config.yaml')
    
    servers = cfg["simulation"]["strategy"]
    
    n_clients = [16, 95]
    languages = ["en", "pt"]
    datasets = ["WiSec", "VeReMi"]

    for client in n_clients:

        for lang in languages:

            for dataset in datasets:

                process_accuracy_delays(dataset=dataset,
                                        n_clients=client,
                                        language=lang,
                                        servers=servers)
    
