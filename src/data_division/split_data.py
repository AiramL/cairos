import os 
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from torch.utils.data  import Subset

from torchvision import (
        datasets, 
        transforms)

from pickle import (
        dump, 
        load)

from utils.torch.sign_dataset import SignDataset
#from utils.visualization.distribution import distribution_plot


def save_matrix_figure(n_classes,
                       n_clients,
                       client_indexes,
                       alpha,
                       dataset,
                       dataset_name,
                       figure_path):
    
    # counter
    matrix = np.zeros((n_classes, n_clients), dtype=int)

    for client_id, indexes in enumerate(client_indexes):
        
        for idx in indexes:
        
            _, class_label = dataset[idx]
            matrix[class_label][client_id] += 1

    # save figure
    plt.figure(figsize=(10, 6))

    sns.heatmap(matrix, annot=True, 
                fmt='d', 
                cmap="YlGnBu", 
                cbar=True)
    
    plt.xlabel("Client")
    plt.ylabel("Class")
    plt.title(f"Clients data distribution (Dirichlet alpha = {alpha}) for {dataset_name}")
    
    plt.savefig(figure_path + ".png", 
                dpi=300, 
                bbox_inches='tight')
    plt.close()


def main(dataset_name:str="CIFAR-10", 
         alpha:float=5.0, 
         n_clients:int=60) -> None:
    
    if dataset_name == "CIFAR-10":
    
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010)),
                                          ])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                  (0.2023, 0.1994, 0.2010)),
                                            ])
        
        train_data = datasets.CIFAR10(root=f'datasets/{dataset_name}', 
                                      train=True, 
                                      download=True, 
                                      transform=transform_train)
        
        test_data = datasets.CIFAR10(root=f'datasets/{dataset_name}', 
                                     train=False, 
                                     download=True, 
                                     transform=transform_test)

    elif dataset_name == "CIFAR-100":
    
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5071, 0.4867, 0.4408), 
                                                                   (0.2675, 0.2565, 0.2761)),
                                          ])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4867, 0.4408), 
                                                                  (0.2675, 0.2565, 0.2761)),
                                            ])
        
        train_data = datasets.CIFAR100(root=f'datasets/{dataset_name}', 
                                       train=True, 
                                       download=True, 
                                       transform=transform_train)
        
        test_data = datasets.CIFAR100(root=f'datasets/{dataset_name}', 
                                      train=False, 
                                      download=True, 
                                      transform=transform_test)


    elif dataset_name == "MNIST":

  
        transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.RandomRotation(10),
                                              transforms.RandomCrop(28, padding=4),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                            ])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((32, 32)),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                            ])

        train_data = datasets.MNIST(root=f'datasets/{dataset_name}', 
                                    train=True, 
                                    download=True, 
                                    transform=transform_train)


        test_data = datasets.MNIST(root=f'datasets/{dataset_name}', 
                                    train=True, 
                                    download=False, 
                                    transform=transform_test)

    elif dataset_name == "FMNIST":

        transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(28, padding=4),  
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.2860,), (0.3530,)) 
                                            ])

        
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((32, 32)),
                                             transforms.Normalize((0.2860,), (0.3530,))
                                            ])

        train_data = datasets.FashionMNIST(root=f'datasets/{dataset_name}', 
                                           train=True, 
                                           download=True, 
                                           transform=transform_train)
        

        test_data = datasets.FashionMNIST(root=f'datasets/{dataset_name}', 
                                          train=False, 
                                          download=True, 
                                          transform=transform_test)

    elif dataset_name == "SIGN":

        transform = transforms.Compose([transforms.ToTensor()])

        # load dataset
        with open("datasets/traffic_signs/datasets/valentynsichkar/traffic-signs-preprocessed/versions/2/data1.pickle","rb") as reader:

            data = load(reader)

        # join data
        x_train = np.concatenate((data['x_train'], 
                                  data['x_validation']), 
                                  axis=0)

        y_train = np.concatenate((data['y_train'], 
                                  data['y_validation']), 
                                  axis=0)
        
        x_test = data['x_test']

        x_train = np.transpose(x_train, 
                               (0, 3, 2, 1))
        
        x_test = np.transpose(x_test, 
                              (0, 3, 2, 1))

        train_data = SignDataset(x_train,
                                 y_train,
                                 transform)
        
        test_data = SignDataset(x_test,
                                data['y_test'],
                                transform)

    else:

        raise ValueError("Dataset not found.")

    n_classes = len(train_data.classes)

    # group indexes by class
    class_indexes = [ [] 
                     for _ in 
                     range(n_classes) ]
    
    for idx, (_, label) in enumerate(train_data):
        
        class_indexes[label].append(idx)

    # apply dirichlet
    client_indexes = [ [] 
                      for _ in 
                      range(n_clients) ]

    for c in range(n_classes):

        class_indices = class_indexes[c]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(np.repeat(alpha, 
                                                    n_clients))

        # verify null distributions
        if proportions.sum() == 0 or np.isnan(proportions.sum()):

            proportions = np.ones(n_clients) / n_clients

        else:

            proportions = proportions / proportions.sum()

        split = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        split_indices = np.split(class_indices, split)

        for i, idxs in enumerate(split_indices):

            client_indexes[i].extend(idxs.tolist())

    # create path
    data_path = f"datasets/{dataset_name}/distributions/nclients_{n_clients}/alpha_{alpha}"
    os.makedirs(data_path, 
                exist_ok=True)
    
    with open(f"{data_path}/indexes", "wb") as writer:
            
        dump(client_indexes,
             writer)
        
    for cid, idxs in enumerate(client_indexes):
        
        client_train = Subset(train_data, idxs)
        
        with open(f"{data_path}/client_{cid}.pkl", "wb") as writer:
            
            dump([client_train,
                  test_data],
                  writer)

    # save data
    #distribution_plot(number_of_clients=n_clients,
    #                  number_of_classes=n_classes,
    #                  dataset_name=dataset_name,
    #                  dataset_object=train_data,
    #                  alpha=alpha)


if __name__ == "__main__":
    
    # parameters
    n_clients = int(sys.argv[1])
    dataset = sys.argv[2]
    alpha = float(sys.argv[3])

    main(dataset_name=dataset,
         alpha=alpha,
         n_clients=n_clients)
