# CAIROS: Controle Adaptativo do aprendIzado fedeRadO em redes Sem fio

Vehicular Federated Learning (VFL) is applied to the training of AI models to ensure user data privacy. However, clients exhibit greater variation in the communication channel than in static scenarios due to high client mobility, exceeding the round response timeout. This reduces system performance, as updates from straggler clients are discarded if they exceed the transmission timeout for the round. This work proposes CAIROS, a strategy for model training in vehicular learning that allows each client to estimate its network and computing conditions through an LSTM model. Based on this estimate, the client decides whether to continue training or to send the calculated parameters early to avoid a timeout. The results show that CAIROS, compared to FedAvg, reduces the incidence of discarded updates due to timer expiration in VFL by up to 38%, increasing the accuracy of the trained models by up to 25%.

# README Structure

- [Organization](#organization)
- [Considered Seals](#considered_seals)
- [Basic Information](#basic_information)
- [Minimum Requirement](#minimum_requirement)
- [Dependencies](#dependencies)
- [Security Concerns](#security_concerns)
- [Requirements](#requirements)
- [Installation](#installation)
- [Minimal Execution](#minimal_execution)
- [Experiments](#experiments)
- [Paper](#paper)
- [LICENSE](#license)

# Organization

Our code has the following structure when cloned from GitHub:

```
├── architectures
│   └── torch
│       ├── custom_models.py
│       ├── flisbee.py
│       ├── implementation.py
│       └── resnet.py
├── config
│   └── config.yaml
├── generate_figures
│   ├── accuracy.py
│   └── efficiency.py
├── LICENSE
├── README.md
├── requirements.txt
├── scripts
│   ├── build
│   │   ├── data.sh
│   │   ├── dependencies.sh
│   │   ├── env.sh
│   │   └── paths.sh
│   ├── run
│   │   ├── baremetal.sh
│   │   ├── client.sh
│   │   ├── docker.sh
│   │   ├── experiments.sh
│   │   ├── jupyter.sh
│   │   ├── processed
│   │   │   ├── accuracy.sh
│   │   │   ├── communication.sh
│   │   │   ├── mobility.sh
│   │   │   └── results.sh
│   │   ├── raw
│   │   │   ├── accuracy.sh
│   │   │   ├── all_accuracy.sh
│   │   │   ├── communication.sh
│   │   │   └── mobility.sh
│   │   ├── server.sh
│   │   ├── test.sh
│   │   └── train_estimator.sh
│   ├── stop
│   │   ├── docker.sh
│   │   ├── flwr.sh
│   │   └── torch
│   │       └── clean.sh
│   └── visualize
│       ├── accuracy.sh
│       ├── animation.sh
│       ├── communication.sh
│       ├── energy.sh
│       ├── mobility.sh
│       └── time2acc.sh
├── src
│   ├── data_division
│   │   └── split_data.py
│   └── federated_learning
│       ├── client
│       │   └── torch
│       │       ├── all_clients.py
│       │       ├── app.py
│       │       ├── client.py
│       │       └── Dockerfile
│       ├── prototype
│       │   ├── client.py
│       │   ├── main.py
│       │   ├── server.py
│       │   └── utils
│       │       ├── distillation.py
│       │       └── load_federated_data.py
│       └── server
│           └── torch
│               ├── app.py
│               ├── Dockerfile
│               └── strategy
│                   ├── fedavg.py
└── utils
    ├── data
    │   ├── get_image_datasets.py
    │   └── get_signs_dataset.py
    ├── epochs_distributions.py
    ├── estimator
    │   ├── architecture.py
    │   ├── data.py
    │   ├── load.py
    │   ├── lstm.py
    │   ├── test.py
    │   └── train.py
    ├── __init__.py
    ├── loader.py
    ├── process
    │   ├── __init__.py
    │   ├── poi.py
    │   └── results
    │       ├── processed
    │       │   ├── accuracy.py
    │       │   ├── aggregate.py
    │       │   ├── communication.py
    │       │   ├── epoch.py
    │       │   └── mobility.py
    │       └── raw
    │           └── communication.py
    ├── torch
    │   ├── load_federated_data.py
    │   └── utils.py
    ├── utils.py
    └── visualization
        ├── accuracy.py
        ├── animation.py
        ├── communication.py
        ├── energy.py
        ├── epoch_delays.py
        ├── legends.py
        └── time2acc.py
```
During its execution, other paths will be created to store the log and results.

# Considered Seals

In the evaluation process, we consider all four seals: Available Artifacts (SeloD), Functional Artifacts (SeloF), Sustainable Artifacts (SeloS), and Reproducible Experiments (SeloR).


# Basic Information 

The system executes the training as exhibited in the figure below.

![Training dynamic in CAIROS](figures/system/dynamic.png)

The mobility and communication models were adapted from [TOFL](https://github.com/AiramL/TimeOptimizedFederatedLearning). Refer to it to get technical details.

# Minimum Requirement

- SO: Ubuntu 22.04.5 LTS
- Cores: 2
- Memory: 4 GB
- Storage: 64 GB

The time values present in the next sessions for installing and executing the system are based on this configuration with a limited number of clients. To fully execute the results with the same parameters as presented in the paper, it might take hours.

# Dependencies

This repository has the following dependencies:

- VirtualBox 7.1.12 (for the VM execution only)
- Git command 2.34.1
- Python3.12
- Conda 25.5.1
- SUMO 1.24.0
- pandas 2.3.1
- numpy 1.26.4
- torch 2.3.0
- torchvision 0.18.0
- matplotlib 3.10.3
- flower 1.7.0
- scikit-learn 1.7.1
- seaborn 0.13.2
- scikit-image 0.25.2

# Security Concerns

Our code only uses data from image datasets and libraries well-known in the literature. Therefore, this code does not impose any risk for the host during its execution.

# Installation

## Virtual Machine

The entire environment was virtualized to facilitate easier execution. You can download the virtual machine image from the following address:
```bash
wget https://gta.ufrj.br/~airam/cairos.ova
```

Load the image into VirtualBox to run the experiments, and execute all commands as the root user.

```bash
user: root
password: SBRC2026
```

When using the provided virtual machine, you can skip directly to the [Experiments](#experiments) Section.

## Baremetal

### Conda (1 minute)

#### Get the script to install miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

#### Change script permissions
```bash
chmod +x Miniconda3-latest-Linux-x86_64.sh
```

#### Execute installation
```bash
./Miniconda3-latest-Linux-x86_64.sh
```
Accept all the conditions and choose the path to install miniconda3, by default, it is located in/root/miniconda3.

#### Activate conda environment

Execute the command below to activate the conda environment:

```bash
source ~/.bashrc
```

### Installing the requirements of our code (2 minutes)

Clone this repository:

```bash
git clone git@github.com:AiramL/cairos.git
```

Install the dependencies with the command below:

```bash 
source scripts/build/dependencies.sh 
```

Create a new conda environment to install necessary packages:

```bash 
source scripts/build/env.sh 
```

Create necessary directories to save the results:

```bash 
source scripts/build/paths.sh 
```

### Generating the mobility and communication data for the experiments (1 minute)

Execute the command to generate random trips with SUMO with the given number of clients:

```bash 
source scripts/run/raw/mobility.sh 
```

Now, we process the raw results to transform them into a pandas dataframe, which will be used by our communiation model:

```bash 
source scripts/run/processed/mobility.sh 
```
The communication model receives as input clients' positions and calculates the throughput. This is executed several times:

```bash 
source scripts/run/raw/communication.sh 
```

We process all the repeated communication results to generate a mean value of users' throughput:

```bash 
source scripts/run/processed/communication.sh 
```

Finally, we train clients' estimator with the throughput data that we have generated previously:

```bash 
source scripts/run/train_estimator.sh 
```

# Minimal Execution

For executing CAIROS, there is a script: 

```bash 
source scripts/run/experiments.sh 
```

The parameters were configured for a minimal execution. 

# Experiments

When executing the script above, we execute a single run of all experiments. To fully generate, it takes hours.

## Experiment 1: Training efficiency

## Experiment 2: Model's performance



## Conclusion 

If we were able to generate the selection and classification data, the test was successful. To reproduce the exact results in the paper, you must change the simulation by copying and pasting the parameters as follows on the [config/config.yaml](config/config.yaml):

```yaml
environment: "cairos"

datasets:
        CIFAR-10: 
                classes: 10
                features: 32,32,3
        MNIST: 
                classes: 10
                features: 32,32,3
        FMNIST:
                classes: 10
                features: 32,32,3
        SIGN: 
                classes: 43
                features: 32,32,3
simulation:
        cars: 60
        mobility:
                distance:
                        x: 1000
                        y: 1000       
                repetitions: 10
        communication:
                repetitions: 30
        speed:
                index: 
                        - 0
                        - 1
                        - 2
                value:
                        - 3.638889 # 13.1km/h
                        - 8.333333 # 30 km/h
                        - 13.88889 # 50 km/h
        federated_learning:
                server:
                        epochs: 50
        base_station:
                range: 1200
                positions: "communication/base_stations.csv"

```

# Paper

[CAIROS: Controle Adaptativo do Aprendizado Federado em Redes Sem Fio](https://www.gta.ufrj.br/ftp/gta/TechReports/SAC26.pdf)

Cite this work as:

```bibtex
@inproceedings{souza2025cairos,
  title={CAIROS: Controle Adaptativo do Aprendizado Federado em Redes Sem Fio},
  author={de Souza, L. A. C., Achir, N., Campista, M. E. M., Costa, L. H. M. K.},
  booktitle={Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos (SBRC)},
  year={2026},
  organization={SBC}
}
```

# LICENSE

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
