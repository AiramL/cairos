import os
import numpy as np
import matplotlib.pyplot as plt
from .lognormal_generator import generate_delay

def generate_lognormal_data(sample_size, mu, sigma, save_path="figures/generated_delays.png"):
    """
    Generates data from a log-normal distribution and saves a histogram of the results.
    
    Parameters:
    - sample_size (int): The number of data points to generate.
    - mu (float): The mean of the underlying normal distribution.
    - sigma (float): The standard deviation of the underlying normal distribution.
    - save_path (str): The file path where the plot will be saved.
    
    Returns:
    - numpy.ndarray: The generated log-normal data.
    """
    
    # 1. Generate the data
    #data = np.random.lognormal(mean=mu, sigma=sigma, size=sample_size)

    data = []

    for _ in range(sample_size):

        data.append(generate_delay(mean=mu, std=sigma))
    
    # 2. Ensure the output directory exists
    dir_name = os.path.dirname(save_path)
    if dir_name:  # Only attempt to create if a directory path is actually provided
        os.makedirs(dir_name, exist_ok=True)
        
    # 3. Create the histogram plot
    plt.figure(figsize=(10, 6))
    
    bins = int(np.sqrt(len(data)))
    

    # Plot the histogram of the generated data
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Generated Data')
    
    # Overlay the theoretical Probability Density Function (PDF)
    x = np.linspace(min(data), max(data), 1000)
    # PDF mathematical formula for log-normal
    pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
    plt.plot(x, pdf, color='red', linewidth=2, label='Theoretical PDF')
    
    # Formatting
    plt.title(f'Log-Normal Distribution\n(n={sample_size}, underlying μ={mu}, underlying σ={sigma})')
    plt.xlabel('Value (e.g., Delay/Latency)')
    plt.ylabel('Density')
    plt.legend()
    #plt.xlim(0,0.2)
    plt.grid(axis='y', alpha=0.3)
    
    # 4. Save the plot to the specified path and close the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close() 
    
    print(f"Successfully generated {sample_size} data points.")
    print(f"Histogram saved to: {save_path}")
    
    return data

# ==========================================
# Example usage:
# ==========================================
if __name__ == "__main__":

    m = 0.054826
    s = 0.014477

    # Generate 10,000 data points with underlying mu=0.5 and underlying sigma=0.8
    my_delays = generate_lognormal_data(
        sample_size=10000, 
        mu=m, 
        sigma=s
    )
    
    # (Optional) You can also save the generated data to a CSV if needed:
    # import pandas as pd
    # pd.DataFrame({"latency": my_delays}).to_csv("figures/generated_delays.csv", index=False)