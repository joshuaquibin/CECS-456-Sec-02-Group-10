# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import hdbscan
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, silhouette_samples

# Load the dataset, skipping the first column
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00484/tripadvisor_review.csv"
df = pd.read_csv(url)


column_names = ['User_ID',
               'galleries',
               'clubs',
               'bars',
               'restaurants',
               'museums',
               'resorts',
               'parks',
               'beaches',
               'theater',
               'churches'
               ]

df.columns = column_names
df = df.drop('User_ID', axis = 1)
df.head()
# 1.    Perform exploratory data analysis (EDA) using data visualization, for example, histogram of the features, boxplot, and apart from this you are encouraged to explore EDA and plot relevant graphs.
#               1.1 Identify the outliers in the dataset
df.boxplot(figsize = (10,8), rot = 90, fontsize= '8', grid = False)
plt.show()

#               1.2 Plot the correlation matrix for the dataset.
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

#               1.3 Plot the graphical distribution for the variables
# df.hist(bins = 10,figsize=(10,10))
# plt.show()

# 2.    Identify the optimal number of clusters in the dataset.
#                2.1.  You may want to compare silhouette and elbow method.

#Elbow
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss,'bx-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#Silhouette
silhouette_scores = []
for i in range(2,11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df)
    silhouette_scores.append(silhouette_score(df,kmeans.labels_))
plt.plot(range(2,11),silhouette_scores,'bx-')
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.show()


# 3.    Use k-means algorithm for creating the clusters.
#               3.1 Interpret each of the clusters in question 3.
c = 0
for i in range(10):
    for j in range(10):
        if j<i:
            c += 1
            km = KMeans(n_clusters=4,
                        init='random',
                        n_init=10,
                        max_iter=300,
                        tol=1e-04,
                        random_state=0)

            y_km = km.fit_predict(np.array(df.iloc[:, [i, j]]))
            colors = ['green', '#34E0A1', '#416788', 'orange']

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)

            for z in range(4):
                label = 'Cluster ' + str(z)

                plt.style.use('seaborn')
                plt.scatter(np.array(df.iloc[:, [i, j]])[y_km == z, 0],
                            np.array(df.iloc[:, [i, j]])[y_km == z, 1],
                            s=50, c=colors[z],
                            marker='o', alpha=0.8,
                            label=label)

            plt.scatter(km.cluster_centers_[:, 0],
                        km.cluster_centers_[:, 1],
                        s=200, marker='o',
                        c='red', label='Centroids')
            plt.legend(scatterpoints=1)
            plt.ylabel(df.columns[j])
            plt.xlabel(df.columns[i])
            plt.title("[" + df.columns[i] + "] [" + df.columns[j] + "]")

plt.tight_layout()
plt.show()

# 4.Use HDBSCAN to perform hierarchical clustering and plot the dendrogram.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
clusterer = hdbscan.HDBSCAN(min_cluster_size=4, metric='euclidean')
clusterer.fit(X_scaled)
df['HDBSCAN_Cluster'] = clusterer.labels_
#
# # Visualize the dendrogram
linkage_matrix = linkage(X_scaled, 'ward')
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.show()
