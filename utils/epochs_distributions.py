import numpy as np

from sys import argv

def generate_epochs_distributions(n_clients=10,
                                  i_epochs=5,
                                  variance=1,
                                  min_epochs=1,
                                  max_epochs=20,
                                  distribution="equal"):

    if distribution == "equal":

        return np.array([ i_epochs for _ in range(n_clients) ])

    elif distribution == "uniform":

        return np.random.randint(low=min_epochs, 
                                 high=max_epochs, 
                                 size=n_clients)
    
    elif distribution == "poison":
        
        return np.random.poison(lam=i_epochs,
                                size=n_clients) + 1
    
    elif distribution == "normal":

        return np.random.normal(loc=i_epochs, 
                                scale=variance, 
                                size=n_clients).round().astype(int)

if __name__ == "__main__":
    
    n_clients    = int(argv[1])
    i_epochs     = int(argv[2])
    distribution = argv[3]

    dist = generate_epochs_distributions(n_clients=n_clients,
                                         i_epochs=i_epochs,
                                         distribution=distribution)

    string = ""

    for item in dist:

        string += f" {item}"
        
    print(string)
