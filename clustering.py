import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# Load Dataset
df = pd.read_csv('Mall_Customers.csv')

# Display Dataset Information
print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Select Annual Income and Spending Score
X = df.iloc[:, [3, 4]].values

# -----------------------------
# Elbow Method
# -----------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

# Save Graph
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')

plt.show()
plt.close()

print("Elbow graph saved as elbow_method.png")

# -----------------------------
# K-Means Clustering
# -----------------------------
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

y_kmeans = kmeans.fit_predict(X)

# Add Cluster Column
df['Cluster'] = y_kmeans

# -----------------------------
# Customer Segmentation Graph
# -----------------------------
plt.figure(figsize=(8, 6))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], label='Cluster 5')

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    marker='X',
    label='Centroids'
)

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

# Save Graph
plt.savefig('customer_segments.png', dpi=300, bbox_inches='tight')

plt.show()
plt.close()

print("Customer segmentation graph saved as customer_segments.png")

# -----------------------------
# Save Clustered Dataset
# -----------------------------
df.to_csv('customer_segments_data.csv', index=False)

print("Clustered dataset saved as customer_segments_data.csv")

# -----------------------------
# Show Save Location
# -----------------------------
print("\nFiles saved in:")
print(os.getcwd())