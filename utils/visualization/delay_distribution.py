import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_delay_distribution(dataset: str,
                            model: str,
                            hardware: str,
                            root: str = 'results/centralized/',
                            subpath: str = '5.0/all_data_fl/0.01',
                            filename: str = 'batch_execution_time',
                            figsize=(8, 5),
                            save_path: str = "figures",
                            show: bool = True):
    
    # 1. Construct the full file path to match:
    # root/$DATASET/subpath/$MODEL/filename

    if hardware == "" or hardware == "inria_1000":

        subpath = "5.0/single_data_0/0.9"

    
    elif hardware != "leme":

        subpath = "5.0/single_data_0/1.0"

    file_dir = os.path.join(root, hardware, 'classification', dataset, subpath, model)
    file_path = os.path.join(file_dir, filename)
    
    # Optional: Automatically handle missing extensions if the exact filename isn't found
    if not os.path.exists(file_path):

        if os.path.exists(file_path + '.txt'):

            file_path += '.txt'

        elif os.path.exists(file_path + '.csv'):

            file_path += '.csv'

        else:

            print(f"Could not find the file at {file_path}")

            return

    # 2. Read the file and skip the first line
    # np.loadtxt is highly efficient for reading numerical arrays and allows skipping rows easily
    try:

        delays = np.loadtxt(file_path, skiprows=1)

    except Exception as e:

        raise RuntimeError(f"Error reading {file_path}: {e}")
        
    n_samples = len(delays)
    if n_samples == 0:
        print("The file contains no data samples after skipping the header.")
        return

    # 3. Calculate the number of bins using the square-root rule
    #n_bins = int(math.ceil(math.sqrt(n_samples)))
    n_bins = 50

    # 4. Set up a clean, professional visualization
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=figsize)

    # Plot the histogram with a Kernel Density Estimate (KDE) line for better data insights
    ax = sns.histplot(
        delays, 
        bins=n_bins, 
        kde=True, 
        color='steelblue', 
        edgecolor='black', 
        alpha=0.7
    )

    # Styling the labels and title
    plt.title(f'Delay Distribution - {dataset.upper()} ({model})', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Execution Time (s)', fontsize=12, labelpad=10)
    plt.ylabel('Frequency (#)', fontsize=12, labelpad=10)
    
    # Remove top and right spines for a cleaner look
    sns.despine(top=True, right=True)
    plt.tight_layout()

    # 5. Save the plot
    if save_path:

        os.makedirs(f"{save_path}/{filename}/{dataset}/{hardware}", exist_ok=True)
        # Create a descriptive output filename
        out_filename = f"hist_{model}.png"
        out_filepath = os.path.join(f"{save_path}/{filename}/{dataset}/{hardware}", out_filename)
        
        # dpi=300 ensures it is high enough quality for publication/reports
        plt.savefig(out_filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {out_filepath}")

    # 6. Show the plot
    if show:
        plt.show()
        
    # Free memory
    plt.close()

if __name__ == '__main__':

    hardware ='inria_1000'

    #for hardware in ['inria', 'leme', "jetson", "raspberry"]:

    for file_name in ["batch_execution_time", "epoch_execution_time", "inference_time"]:

        for dataset in ["CIFAR-10", "MNIST", "SIGN", "CIFAR-100"]:

            for model in ["RESNET10", "CNN", "RESNET18", "RESNET34", "MOBILENETV2", "FLISBEE", "SQUEEZENET", "SHUFFLENET"]:

                plot_delay_distribution(dataset, model, hardware, filename=file_name)
