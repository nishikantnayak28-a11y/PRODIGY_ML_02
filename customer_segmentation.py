import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("Dataset Shape:", df.shape)
print(df.head())

#Feature Selection
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

#Elbow Method
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

#Plotting Elbow Curve
plt.figure(figsize=(8,5))

plt.plot(range(1,11), wcss, marker='o')

plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

plt.show()

#Train Final K-Means Model
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

y_kmeans = kmeans.fit_predict(X)

#Cluster Visualization
plt.figure(figsize=(10,7))

plt.scatter(
    X.iloc[:,0],
    X.iloc[:,1],
    c=y_kmeans,
    s=100
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    marker='X'
)

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.show()

#Printing Clusters Centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)
