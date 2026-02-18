import numpy as np
from scipy.stats import norm

# Generate sample data (2 Gaussian clusters)
np.random.seed(0)
data1 = np.random.normal(5, 1, 100)
data2 = np.random.normal(10, 1, 100)
data = np.concatenate([data1, data2])

# Number of clusters
k = 2
n = len(data)

# Initialize parameters
means = np.random.choice(data, k)
variances = np.random.random(k)
mixing_coeff = np.ones(k) / k

# EM Algorithm
def em_algorithm(data, means, variances, mixing_coeff, iterations=100):

    n = len(data)
    k = len(means)

    for iteration in range(iterations):

        # -------- E STEP --------
        responsibilities = np.zeros((n, k))

        for i in range(k):
            responsibilities[:, i] = mixing_coeff[i] * norm.pdf(data, means[i], np.sqrt(variances[i]))

        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)

        # -------- M STEP --------
        for i in range(k):
            Nk = responsibilities[:, i].sum()

            means[i] = (responsibilities[:, i] * data).sum() / Nk
            variances[i] = (responsibilities[:, i] * (data - means[i])**2).sum() / Nk
            mixing_coeff[i] = Nk / n

    return means, variances, mixing_coeff


# Run EM
final_means, final_variances, final_mixing = em_algorithm(
    data, means, variances, mixing_coeff, iterations=100
)

print("Final Means:", final_means)
print("Final Variances:", final_variances)
print("Final Mixing Coefficients:", final_mixing)
