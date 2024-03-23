import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

plt.figure(figsize=(10, 6))

plt.plot(df['ArraySize'], df['Sequential'], marker='.', label='Sync Time (ms)')
plt.plot(df['ArraySize'], df['Parallel'], marker='.', label='Async Time (ms)')

plt.title('Execution Times')
plt.xlabel('Array Size')
plt.ylabel('Time (ms)')
plt.legend()

plt.show()
