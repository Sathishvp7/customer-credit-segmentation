# Importing the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv('CustomersDetails.csv', index_col='CUST_ID')

# Methods
def get_outlier_details(data, col_name):
    """
    Collects feature's outliers details
    
    Parameters
    ----------
    data : TYPE
        Dataframe.
    col_name : TYPE
        string.

    Returns
    -------
    None.

    """
    data.sort_values(col_name, ascending = True, inplace=True)
    q1 = data[col_name].quantile(0.25)
    q3 = data[col_name].quantile(0.75)
    iqr = q3-q1
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    print(data[col].describe())
    print('min: ', fence_low, ', max:', fence_high) 
    print('--------------------------------------------------------')
    
    
def remove_outlier(data, col_name):
    """
    Removes the outliers

    Parameters
    ----------
    data : TYPE
        Dataframe.
    col_name : TYPE
        string.

    Returns
    -------
    TYPE
        Series.

    """
    data.sort_values(col_name, ascending = True, inplace=True)
    q1 = data[col_name].quantile(0.25)
    q3 = data[col_name].quantile(0.75)
    iqr = q3 - q1
    fence_low  = q1 - (1.5*iqr)
    fence_high = q3 + (1.5*iqr)
    median = data[col].median()
    print(col,'--', fence_low,'--', fence_high,'--', median)
    if (fence_low != 0 and fence_high != 0):
        print('entered')
        data[col] = data.loc[((data[col_name] > fence_low) & 
                              (data[col_name] < fence_high)), col]
        data[col].fillna(median, inplace=True)
    return data[col]


# Data Preprocessing
dataset.fillna(0, inplace=True)

# Identifying and Removing Outliers
for col in dataset.columns:
    get_outlier_details(dataset, col)

for col in dataset.columns:
    dataset[col] = remove_outlier(dataset, col)
    
# Standardizing the data
X = dataset.values
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Finding n_components to maintain : Threshold is 95%
pca = PCA().fit(X)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.axhline(y=0.95, color='r', linestyle='-')
plt.axvline(x=16, color='r', linestyle='-')
plt.title('Pulsar Dataset Explained Variance')
plt.show()

# Feature engineering
pca = PCA(n_components=12, random_state=0)
X = pca.fit_transform(X)


# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, max_iter=1000, init = 'k-means++', random_state = 123)
    kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 10), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 123)
y_kmeans = kmeans.fit_predict(X)

metrics.silhouette_score(X, kmeans.labels_)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'orange', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()