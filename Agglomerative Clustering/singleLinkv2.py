import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Manually create a dataset with values
X = np.array([[39, 61], [73, 9], [28, 33], [50, 85], [63, 14], [78, 65], [15, 29], [40, 89], [27, 73], [51, 4]])

# Create the Agglomerative Clustering model with complete linkage
model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')

# Fit the model to the data
model.fit(X)
labels = model.labels_

# Create a dendrogram
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(X, method='single'))

# Show the dendrogram
plt.title('Dendrogram (Complete-Linkage)')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distances')
plt.show()

# Scatter plot for visualization
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(5):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], s=50, marker='o', color=colors[i], label=f'Cluster {i + 1}')

# Show the scatter plot
plt.title('Complete-Linkage Agglomerative Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
