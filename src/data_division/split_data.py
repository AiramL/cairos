import os 
import sys
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from torch.utils.data  import Dataset
from torchvision import datasets, transforms
from pickle import dump, load
from sklearn.model_selection import train_test_split

class SignDataset(Dataset):

    def __init__(self,
                 data,
                 labels,
                 transform=None):

        self.data = data
        self.labels =  labels
        self.transform =  transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self,
                    idx):

        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:

            x = self.transform(x)

        return x, y

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


def main(dataset_name="CIFAR-10", 
         alpha=0.5, 
         n_clients=5, 
         trPer=0.2):
    
    # download dataset
    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_name == "CIFAR-10":

        dataset = datasets.CIFAR10(root=f'datasets/{dataset_name}', 
                                   train=True, 
                                   download=True, 
                                   transform=transform)

        n_classes = len(torch.unique(torch.tensor(dataset.targets)))

    elif dataset_name == "MNIST":

        dataset = datasets.MNIST(root=f'datasets/{dataset_name}', 
                                 train=True, 
                                 download=True, 
                                 transform=transform)

        n_classes = len(torch.unique(torch.tensor(dataset.targets)))

    elif dataset_name == "FMNIST":

        dataset = datasets.FashionMNIST(root=f'datasets/{dataset_name}', 
                                        train=True, 
                                        download=True, 
                                        transform=transform)

        n_classes = len(torch.unique(torch.tensor(dataset.targets)))

    elif dataset_name == "SIGN":

        # load dataset
        with open("datasets/traffic_signs/datasets/valentynsichkar/traffic-signs-preprocessed/versions/2/data1.pickle","rb") as reader:

            data = load(reader)

        # join data
        x = np.concatenate((data['x_train'], 
                            data['x_validation'], 
                            data['x_test']), 
                            axis=0)

        y = np.concatenate((data['y_train'], 
                            data['y_validation'], 
                            data['y_test']), 
                            axis=0)
        
        n_classes = len(np.unique(y))

        dataset = SignDataset(x,
                              y,
                              transform)

    else:

        raise ValueError("Dataset not found.")

    # group indexes by class
    class_indexes = [ [] 
                     for _ in 
                     range(n_classes) ]
    
    for idx, (_, label) in enumerate(dataset):
        
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
    data_path = f"datasets/{dataset_name}/distributions/nclients_{n_clients}/alpha_{alpha}/"
    os.makedirs(data_path, exist_ok=True)

    for i, idxs in enumerate(client_indexes):
        
        X = []
        Y = []
        
        for idx in idxs:

            image, label = dataset[idx]
            X.append(image.numpy().transpose(1, 2, 0))
            Y.append(label)

        X = np.array(X,
                     dtype=np.float32)
        
        Y = np.array(Y,
                     dtype=np.float32)
        
        x_train, x_test, y_train, y_test = train_test_split(X, 
                                                            Y, 
                                                            test_size=trPer, 
                                                            random_state=42)
        with open(f"{data_path}cliente_{i}.pkl", "wb") as writer:
            
            dump([x_train,
                  x_test,
                  y_train,
                  y_test], 
                  writer)

    # save data
    figure_name = f"figures/data/distributions/{dataset_name}_nclients_{n_clients}_alpha_{alpha}"
    os.makedirs(os.path.dirname(figure_name), 
                exist_ok=True)

    save_matrix_figure(n_classes, 
                       n_clients, 
                       client_indexes, 
                       alpha, 
                       dataset, 
                       dataset_name, 
                       figure_name)


if __name__ == "__main__":
    
    # parameters
    n_clients = int(sys.argv[1])
    dataset = sys.argv[2]
    alpha = float(sys.argv[3])

    main(dataset_name=dataset,
         alpha=alpha,
         n_clients=n_clients)
