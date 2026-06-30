import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==========================================
# CUSTOMER SEGMENTATION USING K-MEANS
# ==========================================

print("=" * 55)
print("CUSTOMER SEGMENTATION USING K-MEANS")
print("=" * 55)

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Display Dataset
print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

# Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print("\nSelected Features")
print(X.head())

# ==========================================
# ELBOW METHOD
# ==========================================

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,6))
plt.plot(range(1,11), wcss, marker='o', linewidth=2)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.savefig("elbow_method.png")
plt.show()

# ==========================================
# TRAIN K-MEANS MODEL
# ==========================================

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

y_pred = kmeans.fit_predict(X)

df["Cluster"] = y_pred

# ==========================================
# PRINT CLUSTER INFORMATION
# ==========================================

print("\nCluster Count")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster Centers")
print(kmeans.cluster_centers_)

# ==========================================
# CUSTOMER SEGMENTATION GRAPH
# ==========================================

plt.figure(figsize=(10,8))

colors = ['blue', 'green', 'orange', 'purple', 'brown']

cluster_names = [
    "Average Customers",
    "High Income - High Spending",
    "Low Income - High Spending",
    "High Income - Low Spending",
    "Low Income - Low Spending"
]

for i in range(5):
    plt.scatter(
        X[y_pred == i]["Annual Income (k$)"],
        X[y_pred == i]["Spending Score (1-100)"],
        s=70,
        color=colors[i],
        label=cluster_names[i]
    )

# Plot Centroids
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    color="red",
    marker="X",
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(loc="best")
plt.grid(True)

plt.savefig("customer_clusters.png")
plt.show()

print("\nTask 2 Completed Successfully!")
