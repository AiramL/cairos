import os 
import sys
import copy
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from torch.utils.data import Subset
from torchvision import datasets, transforms
from pickle import dump, load

from utils.torch.sign_dataset import SignDataset
# from utils.visualization.distribution import distribution_plot


def save_matrix_figure(n_classes, n_clients, client_indexes, alpha, dataset, dataset_name, figure_path):
    matrix = np.zeros((n_classes, n_clients), dtype=int)
    for client_id, indexes in enumerate(client_indexes):
        for idx in indexes:
            _, class_label = dataset[idx]
            matrix[class_label][client_id] += 1

    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap="YlGnBu", cbar=True)
    plt.xlabel("Client")
    plt.ylabel("Class")
    plt.title(f"Clients data distribution (Dirichlet alpha = {alpha}) for {dataset_name}")
    plt.savefig(figure_path + ".png", dpi=300, bbox_inches='tight')
    plt.close()


def get_dirichlet_distribution(dataset, n_classes, n_clients, alpha):
    """
    Groups data by class and applies Dirichlet distribution to partition
    indices among clients mutually exclusively.
    """
    class_indexes = [[] for _ in range(n_classes)]
    
    # Group indexes by class
    for idx, (_, label) in enumerate(dataset):
        class_indexes[label].append(idx)

    client_indexes = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        class_indices = class_indexes[c]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))

        # Verify null distributions
        if proportions.sum() == 0 or np.isnan(proportions.sum()):
            proportions = np.ones(n_clients) / n_clients
        else:
            proportions = proportions / proportions.sum()

        split = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        split_indices = np.split(class_indices, split)

        for i, idxs in enumerate(split_indices):
            client_indexes[i].extend(idxs.tolist())

    return client_indexes


def extract_client_dataset(dataset, indices):
    """
    Creates an isolated Dataset object for a specific client to save memory/storage.
    Prevents serializing the entire dataset structure attached to standard torch Subsets.
    """
    client_ds = copy.copy(dataset)
    
    # Standard Torchvision datasets (CIFAR, MNIST, etc.) usually use 'data' and 'targets'
    if hasattr(client_ds, 'data') and hasattr(client_ds, 'targets'):
        # Handle features (data)
        if isinstance(client_ds.data, np.ndarray) or torch.is_tensor(client_ds.data):
            client_ds.data = client_ds.data[indices]
        else: # Fallback for lists
            client_ds.data = [client_ds.data[i] for i in indices]
            
        # Handle labels (targets)
        if isinstance(client_ds.targets, list):
            client_ds.targets = [client_ds.targets[i] for i in indices]
        elif isinstance(client_ds.targets, np.ndarray) or torch.is_tensor(client_ds.targets):
            client_ds.targets = client_ds.targets[indices]
            
    # Handle custom datasets like SignDataset (using 'x' and 'y')
    elif hasattr(client_ds, 'x') and hasattr(client_ds, 'y'):
        client_ds.x = client_ds.x[indices]
        client_ds.y = client_ds.y[indices]
        
    return client_ds


def main(dataset_name:str="CIFAR-10", 
         alpha:float=5.0, 
         n_clients:int=60,
         split_test_data:bool=False) -> None:
    
    if dataset_name == "CIFAR-10":
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                  (0.2023, 0.1994, 0.2010))])
        
        train_data = datasets.CIFAR10(root=f'datasets/{dataset_name}', train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR10(root=f'datasets/{dataset_name}', train=False, download=True, transform=transform_test)

    elif dataset_name == "CIFAR-100":
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5071, 0.4867, 0.4408), 
                                                                   (0.2675, 0.2565, 0.2761))])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4867, 0.4408), 
                                                                  (0.2675, 0.2565, 0.2761))])
        
        train_data = datasets.CIFAR100(root=f'datasets/{dataset_name}', train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR100(root=f'datasets/{dataset_name}', train=False, download=True, transform=transform_test)

    elif dataset_name == "MNIST":
        transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.RandomRotation(10),
                                              transforms.RandomCrop(28, padding=4),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((32, 32)),
                                             transforms.Normalize((0.1307,), (0.3081,))])

        train_data = datasets.MNIST(root=f'datasets/{dataset_name}', train=True, download=True, transform=transform_train)
        test_data = datasets.MNIST(root=f'datasets/{dataset_name}', train=False, download=True, transform=transform_test)

    elif dataset_name == "FMNIST":
        transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(28, padding=4),  
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.2860,), (0.3530,))])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((32, 32)),
                                             transforms.Normalize((0.2860,), (0.3530,))])

        train_data = datasets.FashionMNIST(root=f'datasets/{dataset_name}', train=True, download=True, transform=transform_train)
        test_data = datasets.FashionMNIST(root=f'datasets/{dataset_name}', train=False, download=True, transform=transform_test)

    elif dataset_name == "SIGN":
        transform = transforms.Compose([transforms.ToTensor()])

        with open("datasets/traffic_signs/datasets/valentynsichkar/traffic-signs-preprocessed/versions/2/data1.pickle","rb") as reader:
            data = load(reader)

        x_train = np.concatenate((data['x_train'], data['x_validation']), axis=0)
        y_train = np.concatenate((data['y_train'], data['y_validation']), axis=0)
        x_test = data['x_test']

        x_train = np.transpose(x_train, (0, 3, 2, 1))
        x_test = np.transpose(x_test, (0, 3, 2, 1))

        train_data = SignDataset(x_train, y_train, transform)
        test_data = SignDataset(x_test, data['y_test'], transform)

    else:
        raise ValueError("Dataset not found.")

    # Infer the number of classes safely
    if hasattr(train_data, 'classes'):
        n_classes = len(train_data.classes)
    else:
        # Fallback for custom datasets where .classes might not be explicitly defined
        n_classes = len(np.unique([label for _, label in train_data]))

    print(f"Generating train distribution for {n_clients} clients...")
    train_client_indexes = get_dirichlet_distribution(train_data, n_classes, n_clients, alpha)
    
    if split_test_data:
        print("Generating test Dirichlet distribution...")
        test_client_indexes = get_dirichlet_distribution(test_data, n_classes, n_clients, alpha)

    # create path
    data_path = f"datasets/{dataset_name}/distributions/nclients_{n_clients}/alpha_{alpha}"
    os.makedirs(data_path, exist_ok=True)
    
    # Save indexes for reference
    with open(f"{data_path}/train_indexes.pkl", "wb") as writer:
        dump(train_client_indexes, writer)
    if split_test_data:
        with open(f"{data_path}/test_indexes.pkl", "wb") as writer:
            dump(test_client_indexes, writer)
        
    print(f"\nSaving client data (Split Test Data mode: {split_test_data})...")
    for cid in range(n_clients):
        # Slice train data
        client_train = extract_client_dataset(train_data, train_client_indexes[cid])
        
        # Scenario check: Shared Global Test vs Dirichlet Partitioned Test
        if split_test_data:
            client_test = extract_client_dataset(test_data, test_client_indexes[cid])
        else:
            client_test = test_data 
            
        print(f"Client {cid:02d} | Train Size: {len(client_train):5d} | Test Size: {len(client_test):5d}")
        
        with open(f"{data_path}/client_{cid}.pkl", "wb") as writer:
            dump([client_train, client_test], writer)

    print("\nData successfully generated and saved!")

if __name__ == "__main__":
    n_clients = int(sys.argv[1])
    dataset = sys.argv[2]
    alpha = float(sys.argv[3])
    
    # Optional 4th argument to trigger test data splitting (e.g., True or False). Defaults to False.
    split_test = True
    
    if len(sys.argv) > 4:

        split_test = sys.argv[4].lower() in ['false', '0', 'f', 'n', 'no']
    
    main(dataset_name=dataset, alpha=alpha, n_clients=n_clients, split_test_data=split_test)
