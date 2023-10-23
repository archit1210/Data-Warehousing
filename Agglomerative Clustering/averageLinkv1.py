import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Read the dataset
dataset = pd.read_csv("Mall_Customers.csv")

# Extract the relevant columns
X = dataset.iloc[:, [3, 4]].values

# Create the Agglomerative Clustering model with complete linkage
model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')

# Fit the model to the data
model.fit(X)
labels = model.labels_

# Create a dendrogram
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(X, method='complete'))

# Show the dendrogram
plt.title('Dendrogram (Complete-Linkage)')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Scatter plot for visualization
plt.figure(figsize=(8, 6))
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], s=50, marker='o', color='red', label='Cluster 1')
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], s=50, marker='o', color='blue', label='Cluster 2')
plt.scatter(X[labels == 2, 0], X[labels == 2, 1], s=50, marker='o', color='green', label='Cluster 3')
plt.scatter(X[labels == 3, 0], X[labels == 3, 1], s=50, marker='o', color='purple', label='Cluster 4')
plt.scatter(X[labels == 4, 0], X[labels == 4, 1], s=50, marker='o', color='orange', label='Cluster 5')

# Show the scatter plot
plt.title('Complete-Linkage Agglomerative Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
