import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

# Read CSV file
df = pd.read_csv("execution_times.csv")

# Fit a polynomial of degree 2 to TimeGlobal
coeffs_global = np.polyfit(df['ArrayLength'], df['TimeGlobal'], 2)
fit_poly_global = Polynomial(coeffs_global)
fit_values_global = fit_poly_global(df['ArrayLength'])

# Fit a polynomial of degree 2 to TimeCPU
coeffs_cpu = np.polyfit(df['ArrayLength'], df['TimeCPU'], 2)
fit_poly_cpu = Polynomial(coeffs_cpu)
fit_values_cpu = fit_poly_cpu(df['ArrayLength'])

# Plotting with thicker lines and larger fonts
plt.figure(figsize=(10, 6))
plt.plot(df['ArrayLength'], fit_values_global, label='CPU', linestyle='--', color='blue', linewidth=2)
plt.plot(df['ArrayLength'], fit_values_cpu, label='GPU', linestyle='--', color='orange', linewidth=2)

# Add labels and title with larger fonts
plt.xlabel('Array Length', fontsize=15)
plt.ylabel('Execution Time (ms)', fontsize=15)
plt.title('Relationship between Execution Time and Array Length with Trendlines', fontsize=14)

# Increase legend font size
plt.legend(fontsize=10)

# Display the plot
plt.show()
