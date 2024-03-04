import pandas as pd
import matplotlib.pyplot as plt

# Read the timing results from the CSV file
df = pd.read_csv("Part1_times.csv")

# Plot the timing results
plt.figure(figsize=(10, 6))

# Plot time taken for coalesced operation
plt.plot(df['Block Size'], df['Time (Coalesced)'], label='Coalesced', marker='')

# Plot time taken for non-coalesced operation
plt.plot(df['Block Size'], df['Time (Not Coalesced)'], label='Not Coalesced', marker='')

plt.ylim(0, 15)

# Set plot title and labels
plt.title('Comparison of Coalesced vs Not Coalesced Timing')
plt.xlabel('Block Size')
plt.ylabel('Time (ms)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
