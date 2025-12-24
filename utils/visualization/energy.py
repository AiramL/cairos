import matplotlib.pyplot as plt

from numpy import mean, std
from pickle import load

from .legends import legends_dicts

def plot_energy(dictionary, 
                PLOT=False, 
                language="en"):
    
    plt.figure(figsize=(14, 10))

    x_axis = [ x*10 
              for x in dictionary[list(dictionary.keys())[0]].keys() ]

    legends = legends_dicts[language]

    for server in dictionary.keys():
        
        means = []
        stds = []
        
        for err in dictionary[server].keys():
            
            means.append(mean(dictionary[server][err],axis=0))
            stds.append(std(dictionary[server][err],axis=0))
            
            
        plt.errorbar(x_axis, 
                     [ 100*x for x in means ], 
                      yerr=[ 100*x for x in stds ], 
                      capsize=3, 
                      label=legends[server])

    if language == "en":        

        plt.xlabel("Clients' Error Rate (%)", fontsize=16)
        plt.ylabel("Training Efficiency (%)", fontsize=16)

    elif language == "pt":
        
        plt.xlabel("Taxa de Erro por Cliente (%)", fontsize=16)
        plt.ylabel("Eficiência de Treinamento (%)", fontsize=16)


    plt.xticks(x_axis, 
               [str(x) for x in x_axis])

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=20)
    plt.savefig("figures/training_efficiency_"+language+".png",
                dpi=300,
                bbox_inches='tight')
    
   
    

    if PLOT:
        plt.show()


if __name__ == "__main__":

    with open("results/energy","rb") as reader:
        
        dictionary = load(reader)

    plot_energy(dictionary)
