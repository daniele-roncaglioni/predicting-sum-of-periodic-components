import numpy as np
from matplotlib import pyplot as plt

n_samples = 100
# Standard deviation of the increments
increment_std = 0.01  # Adjust as needed to control the overall noise level

# Generate incremental steps
increments = np.random.normal(0, increment_std, n_samples)

# Generate the random walk (Brownian noise) by cumulatively summing the increments
brownian_noise = np.cumsum(increments)

# Normalize or scale the Brownian noise to match a specific standard deviation, if needed
target_std = 0.1
actual_std = np.std(brownian_noise)
brownian_noise_scaled = brownian_noise * (target_std / actual_std)

plt.plot(range(len(brownian_noise_scaled)), brownian_noise_scaled)
plt.show()
