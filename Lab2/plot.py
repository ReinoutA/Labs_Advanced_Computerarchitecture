import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV file into a DataFrame
df = pd.read_csv('execution_times.csv')

# Plot the data
plt.figure(figsize=(10, 6))

# Calculate linear regression lines
coeff_cpu = np.polyfit(df['Array Size'], df['CPU'], 1)
trendline_cpu = np.polyval(coeff_cpu, df['Array Size'])

coeff_gpu_a = np.polyfit(df['Array Size'], df['GPU_A'], 1)
trendline_gpu_a = np.polyval(coeff_gpu_a, df['Array Size'])

coeff_gpu_r = np.polyfit(df['Array Size'], df['GPU_R'], 1)
trendline_gpu_r = np.polyval(coeff_gpu_r, df['Array Size'])

# Plot trendlines only
plt.plot(df['Array Size'], trendline_cpu, label='CPU Time (Trendline)')
plt.plot(df['Array Size'], trendline_gpu_a, label='GPU (Atomic) Time (Trendline)')
plt.plot(df['Array Size'], trendline_gpu_r, label='GPU (Reduction) Time (Trendline)')

plt.title('Execution Times vs Array Size')
plt.xlabel('Array Size')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)
plt.show()
