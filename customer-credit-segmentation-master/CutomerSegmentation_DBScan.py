# Importing the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import metrics

# Initializing the dataset
dataset = pd.read_csv('CustomersDetails.csv', index_col='CUST_ID')


# Data Preprocessing
dataset.fillna(0, inplace=True)

# Finding and removing highly corrleated feature
correlation = dataset.corr().abs()
upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

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
plt.axvline(x=10, color='r', linestyle='-')
plt.title('Pulsar Dataset Explained Variance')
plt.show()

# Feature engineering
pca = PCA(n_components=11, random_state=0)
X = pca.fit_transform(X)

# Model Fitting
dbscan = DBSCAN(eps=5, min_samples=4)   
model = dbscan.fit(X)

y_pred = dbscan.fit_predict(X)

labels = model.labels_

# Identifying Core Points
core_ponits = np.zeros_like(labels, dtype=bool)
core_ponits[dbscan.core_sample_indices_] = True


# Calculating number of clusters
n_cluster = len(set(labels)) - (1 if -1 in labels else 0)

print(metrics.silhouette_score(X, labels))

# Visualising the clusters
plt.scatter(X[:,0], X[:,1],c=y_pred, cmap='Paired')
plt.title('Clusters of customers')
plt.show()



