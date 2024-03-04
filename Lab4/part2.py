import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('execution_times.csv')
array_sizes = df['ArraySize']
global_memory_times = df['GlobalMemoryTime(ms)']
shared_memory_times = df['SharedMemoryTime(ms)']
constant_memory_times = df['ConstantMemoryTime(ms)']

plt.figure(figsize=(10, 6))
plt.plot(array_sizes, global_memory_times, label='Global Memory')
plt.plot(array_sizes, shared_memory_times, label='Shared Memory')
plt.plot(array_sizes, constant_memory_times, label='Constant Memory')

plt.xlabel('Array Size')
plt.ylabel('Execution Time (ms)')
plt.title('Execution Time vs Array Size for Different Memory Types')
plt.legend()
plt.grid()

plt.show()
