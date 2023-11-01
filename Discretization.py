#data d and v

import pandas as pd  
import matplotlib.pyplot as plt 

data = {'Age': [22, 35, 45, 28, 32, 50, 28, 40, 60, 55]}  
df = pd.DataFrame(data)  
num_bins = 3  

df['Age_discretized'] = pd.cut(df['Age'], bins=num_bins, labels=False)  
plt.figure(figsize=(8, 6))  

plt.hist(df['Age_discretized'], bins=num_bins, edgecolor='black', alpha=0.7) 

plt.xlabel('Discretized Age Bins')  
plt.ylabel('Frequency')  

plt.title('Histogram of Discretized Age Data')  
plt.xticks(range(num_bins), [f'Bin {i}' for i in range(num_bins)]) 

plt.show()  
print(df)
