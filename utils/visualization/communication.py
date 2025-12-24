from pickle import load
import matplotlib.pyplot as plt
from numpy import mean, std

from utils.utils import load_config 
from .legends import legends_dicts

def selection_error_plot(file_path="results/client_selection/",
                         model_size="model_size500", 
                         epochs=100,
                         PLOT=False, 
                         n_executions=10, 
                         language="pt",
                         servers = ["random",
                                    "m_fastest",
                                    "tofl_oracle",
                                    "tofl_estimator_dl",
                                    "tofl_estimator_m_fastest"]):
    
    plt.figure(figsize=(14, 10))
    

    legends = legends_dicts[language]

    if language == "en":
        
        plt.xlabel("Selected Clients (#)", fontsize=16)
        plt.ylabel("Total Training Time (s)", fontsize=16)

    elif language == "pt":

        plt.xlabel("Quantidade de Clientes Selecionados (#)", fontsize=16)
        plt.ylabel("Tempo Total de Treinamento (s)", fontsize=16)


    results = { server : [ ] 
               for server in servers }
    
    for server in servers:

        for dataset in range(n_executions):
        
            with open(file_path+"model_"+server+model_size+"_dataset_"+str(dataset),"rb") as loader:
            
                result_list = load(loader)
                results[server].append(result_list)
    
    for server in servers:
        
        m = mean(results[server],axis=0)
        s = std(results[server],axis=0)
        x = [ index for index in range(1,1+len(m)) ]
        plt.errorbar(x, 
                     m, 
                     yerr=s, 
                     capsize=3, 
                     label=legends[server])

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)

    plt.savefig("figures/communication_"+model_size[1:]+"_"+language+".png",
                dpi=300,
                bbox_inches='tight')

    if PLOT:
        plt.show()

if __name__ == "__main__":

    cfg = load_config('config/config.yaml')
    
    sizes = cfg["simulation"]["model"]["size"]
    speeds = cfg["simulation"]["speed"]["index"]
    servers = cfg["simulation"]["strategy"]
    repetitions = cfg["simulation"]["mobility"]["repetitions"]
    epochs = cfg["simulation"]["federated_learning"]["server"]["epochs"]    

    for lg in ["pt", "en"]:

        for speed in speeds:

            for model_size in sizes:

                selection_error_plot(f"results/client_selection/speed{speed}/", 
                                     f"_size_{model_size}", 
                                     servers=servers,
                                     epochs=epochs,
                                     n_executions=repetitions,
                                     language=lg)
        
