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
- [FAQ](#faq)

# Organization

Our code has the following structure when cloned from GitHub:

```
├── architectures
│   └── torch
│       ├── custom_models.py
│       ├── implementation.py
│       └── resnet.py
├── communication
│   └── base_stations.csv
├── config
│   └── config.yaml
├── figures
│   └── system
│       └── dynamic.png
├── generate_figures
│   ├── accuracy_error_bar.py
│   ├── accuracy_line_error.py
│   ├── efficiency_error.py
│   └── motivation.py
├── LICENSE
├── README.md
├── requirements.txt
├── scripts
│   ├── build
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
│   │   │   ├── communication.sh
│   │   │   ├── mobility.sh
│   │   │   └── results.sh
│   │   ├── raw
│   │   │   ├── communication.sh
│   │   │   └── mobility.sh
│   │   ├── sbrc_experiments.sh
│   │   ├── server.sh
│   │   └── train_estimator.sh
│   ├── stop
│   │   ├── docker.sh
│   │   ├── flwr.sh
│   │   └── torch
│   │       └── clean.sh
│   └── visualize
│       ├── communication.sh
│       └── mobility.sh
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
│       └── server
│           └── torch
│               ├── app.py
│               ├── Dockerfile
│               └── strategy
│                   └── fedavg.py
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
    │       │   ├── aggregate.py
    │       │   ├── communication.py
    │       │   └── mobility.py
    │       └── raw
    │           └── communication.py
    ├── torch
    │   ├── load_federated_data.py
    │   └── utils.py
    └── utils.py
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

Expected output: 

```bash 
PHASE 1 -> Generating the grid topology
Success.
....... -> Copy netfile to the current directory
....... -> Generate continuous rerouters
....... -> Generating random flows for JTRROUTER
....... -> Run SUMO for configuration Krauss 5 cars / iteration 0
....... -> Generate tr file for configuration 5 cars / iteration 0
One or more coordinates are negative, some applications might need strictly positive values. To avoid this use the option --shift.
```

Now, we process the raw results to transform them into a pandas dataframe, which will be used by our communiation model:

```bash 
source scripts/run/processed/mobility.sh 
```

Expected output: 

```bash
process finished 
```

The communication model receives as input clients' positions and calculates the throughput. This is executed several times:

```bash 
source scripts/run/raw/communication.sh 
```

Expected output: 

```bash 
processing mobility file  0
processing mobility file  2
processing mobility file  1
index  0
index  0
index  0
[1]   Done                    python -m utils.process.results.raw.communication $execution $speed $index
[2]-  Done                    python -m utils.process.results.raw.communication $execution $speed $index
[3]+  Done                    python -m utils.process.results.raw.communication $execution $speed $index
```

We process all the repeated communication results to generate a mean value of users' throughput:

```bash 
source scripts/run/processed/communication.sh 
```

Expected output: 

```bash 
processing file  2
processing file  0
processing file  1
processing finished
processing finished
processing finished
[1]   Done                    python -m utils.process.results.processed.communication $speed $index
[2]-  Done                    python -m utils.process.results.processed.communication $speed $index
[3]+  Done                    python -m utils.process.results.processed.communication $speed $index
```

Finally, we train clients' estimator with the throughput data that we have generated previously:

```bash 
source scripts/run/train_estimator.sh 
```

Expected output: 

```bash
torch.Size([7712, 5, 1]) torch.Size([7712, 5, 1])
torch.Size([3797, 5, 1]) torch.Size([3797, 5, 1])
Epoch 0: train RMSE 8.0887, test RMSE 6.7943
Epoch 10: train RMSE 2.1793, test RMSE 2.1151
tensor([[37.7809],
        [39.3960],
        [39.1443],
        ..., 
```

# Minimal Execution (20 minutes)

For executing CAIROS, there is a script: 

```bash 
source scripts/run/experiments.sh 
```

Expected output: 

```bash
Starting FL training with torch server at 127.0.0.1:8081, 5 clients executing 5 local epochs, for 3 rounds, selecting 5 clients to fit the RESNET10 model for 5 local epochs on the dataset CIFAR-10 using a datadistribution with alpha equals to 5.0, a maximun timeout of 50 seconds, using the cairos_pe strategy, the equal local epochs distribution, and speed index 0.
Creating paths
Files already downloaded and verified
.
.
.
INFO :      [ROUND 3]
INFO :      configure_fit: strategy sampled 5 clients (out of 5)
INFO :
INFO :      Received: train message a133aafe-4557-4681-b414-2974a8a4e1df
INFO :
INFO :      Received: train message c366c5c8-adbd-4f1a-b981-f1dda7c9558b
INFO :
INFO :      Received: train message fb8564f8-9fb3-438c-8de0-c30f288f0d8f
INFO :
INFO :      Received: train message ea3f7a7c-c5ce-4e6c-a1a6-4a4c2ae3bbbd
INFO :
INFO :      Received: train message fbe3c91d-c34f-423a-a0ef-99506fe836dd
INFO :      Sent reply
INFO :      Sent reply
INFO :      Sent reply
INFO :      Sent reply
INFO :      Sent reply
.
.
.
INFO :      Received: reconnect message 499f07f5-b828-4306-843c-ef6cf95861b4
INFO :      Disconnect and shut down
INFO :      Disconnect and shut down
INFO :      Disconnect and shut down
INFO :      Disconnect and shut down
```

The parameters were configured for a minimal execution. 

# Experiments

When executing the script above, we execute a single run of all experiments. To fully generate, it takes hours.

## Experiment 1: Training efficiency

To visualize the resource efficiency during training, you can generate the visualization with the command below:

```python
python -m generate_figures.efficiency_error
```

Expected output: 

```bash 
file: figures/bar_efficiency_avg_iepochs_5_n_selec_5_pt.png
```

## Experiment 2: Model's performance

To visualize the evolution of test accuracy, you can generate the visualization with the command below:

```python
python -m generate_figures.accuracy_line_error
```

Expected output: 

```bash
file: figures/accuracy_line_timeout_50_n_selec_10_pt_avg_std.png 
```

To visualize the final accuracy varying the timeout, you can generate the visualization with the command below:

```python
python -m generate_figures.accuracy_error_bar
```

Expected output: 

```bash
file: figures/bar_accuracy_iepochs_5_n_selec_5_pt_avg_std.png 
```

## Conclusion 

If we were able to generate the selection and classification data, and visualize the three figures, the test was successful. To reproduce the exact results in the paper, you must change the simulation by copying and pasting the parameters as follows on the [config/config.yaml](config/config.yaml):

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

# FAQ

## My training is too slow, what should I do? 

Verify if your machine has a GPU. If it does not have a GPU, rebuild the torch packges using the following lines:

```
torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
```
