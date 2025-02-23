import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# Example: Load your customer dataset
data = pd.read_csv('shopping_trends_updated.csv')

# Let's assume your dataset has columns: 'age', 'income', 'purchase_frequency', 'total_spend'
features = ['Age', 'Purchase Amount (USD)', 'Review Rating']

X = data[features]

# Check for missing values
print(X.isnull().sum())

# Fill missing values or drop rows if necessary (this example drops missing rows)
X.dropna(inplace=True)

# Feature Scaling: Standardize features to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):  # silhouette score needs at least 2 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    cluster_labels = kmeans.labels_
    score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis For Optimal K')
plt.show()

# Choose K based on the previous analysis
optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the original DataFrame for further analysis
data['Cluster'] = cluster_labels

# Group the data by the assigned clusters and compute summary statistics
cluster_summary = data.groupby('Cluster')[features].mean()
print(cluster_summary)

from sklearn.decomposition import PCA

# Reduce dimensions to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a scatter plot of the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Customer Clusters Visualized with PCA')
plt.colorbar(label='Cluster')
plt.show()
