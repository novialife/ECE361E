import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
# Benchmark runtime: 131.90632581710815

data = open('log_bodytrack.txt', 'r').readlines()

import pandas as pd
columns = data[0].split()
df = pd.DataFrame(columns=columns)
for line in data[1:]:
    df = df.append(pd.Series(line.split(), index=columns), ignore_index=True)

# Change time from unix timestamp to datetime
df['time'] = pd.to_datetime(df['time'], unit='s')

# Round time to nearest 10 milliseconds
df['time'] = df['time'].dt.round('1ms')

# Round all the numeric columns to 2 decimal places
for col in df.columns[1:]:
    df[col] = df[col].astype(float).round(4)

# Make a plot with subplots for each temperature
fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10, 10))

for i, temp in enumerate(['temp4', 'temp5', 'temp6', 'temp7']):
    axes[i].plot(df['time'], df[temp], 'o-', label=temp)
    axes[i].set_ylabel('Temperature [°C]')
    axes[i].legend()
    axes[i].grid()

axes[3].set_xlabel('Time')
fig.tight_layout()
axes[0].set_title('Core Temperatures')
fig.savefig

plt.plot(df['time'], df['W'], label='W')
plt.xlabel('Time')
plt.ylabel('W')
plt.title('Power Consumption')
plt.legend()
plt.grid()
plt.show()

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10, 10))

for i, usage in enumerate(['usage_c4', 'usage_c5', 'usage_c6', 'usage_c7']):
    axes[i].plot(df['time'], df[usage]*100, 'o-', label=usage)
    axes[i].set_ylabel('Core Utilization [%]')
    axes[i].legend()
    axes[i].grid()

axes[3].set_xlabel('Time')
axes[0].set_title('Core Utilization')
fig.tight_layout()
plt.grid()
plt.legend()
plt.title('Core Utilization')
plt.show()

# Avergae power consumption
average_power = df['W'].mean()

# Make a new column containing the maximum value of the columns temp4, temp5, temp6, temp7 for each row
df['max_temp'] = df[['temp4', 'temp5', 'temp6', 'temp7']].max(axis=1)

# Mean max temperature
average_max_temp = df['max_temp'].mean()

# Max max temperature
max_max_temp = df['max_temp'].max()

print('Average power consumption: {:.2f} W'.format(average_power))
print('Average max temperature: {:.2f} °C'.format(average_max_temp))
print('Max max temperature: {:.2f} °C'.format(max_max_temp))

# Calculate the energy consumption in Joules as the sum of the power consumption in Watts multiplied by the sampling interval in seconds
#energy_consumption = (df['W'] * 10).sum() # Dont know the sampling interval yet