import numpy as np
import scipy.stats as stats

def get_lognormal_mean(mean, sigma):

    return  np.log(mean**2 / np.sqrt((sigma + mean**2)))


def get_lognormal_std(mean, sigma):

    return  np.sqrt((np.log(1 + (sigma / mean**2))))

def generate_delay(mean=0.054826, std=0.014477, lower_bound=0.00178, upper_bound=0.219849):
    """
    Generates log-normal distributed data points within specified bounds.

    Parameters:
    - sample_size: Number of data points to generate.
    - mu: Mean of the underlying normal distribution.
    - sigma: Standard deviation of the underlying normal distribution.
    - lower_bound: Minimum value for the generated data.
    - upper_bound: Maximum value for the generated data.

    Returns:
    - data: Array of generated log-normal data points.
    """
    mu = get_lognormal_mean(mean, std)
    sigma = get_lognormal_std(mean, std)

    # 1. Create the standard log-normal distribution object
    # (SciPy uses 's' for sigma and 'scale' for exp(mu))
    dist = stats.lognorm(s=sigma, scale=np.exp(mu))

    # 2. Find the Cumulative Distribution Function (CDF) values at your bounds
    cdf_lower = dist.cdf(lower_bound)
    cdf_upper = dist.cdf(upper_bound)

    # 3. Generate random probabilities strictly between those two thresholds
    random_probs = np.random.uniform(cdf_lower, cdf_upper, size=1)

    # 4. Map the probabilities directly back to log-normal values using the 
    # Percent Point Function (PPF), which is the inverse of the CDF.
    data = dist.ppf(random_probs)

    return data[0]

if __name__ == "__main__":

    m = 0.00287
    s = 0.0023

    generate_delay_data = generate_delay(mean=m, std=s)
    print(generate_delay_data)