from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

def euclidean_cluster(n_cluster, X, out_dir="fig/", threshold=0):
    c_type = "euclidean_distance"
    X = X.reshape((n_snaps,-1))
    #clustering = AgglomerativeClustering(n_clusters=n_cluster, compute_full_tree=True).fit(X)
    Z = linkage(X, method='ward')
    clustering = fcluster(Z,t=threshold , criterion='distance')
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(Z, color_threshold=threshold)
    plt.savefig(out_dir + "dendrogram_" + c_type + ".png")
    plt.show()
    plt.close()
    return clustering, Z
    # The linkage matrix shows how the clustering is done, it has n-1 rows and 4 columns, each row shows which clusters are 
    # getting combined their distance and the numbero of observations in the new cluster, the new cluster label is the sum
    # of the labels of the 2 cluster composing it ence a cluster with a label lower than n is a cluster with a single observation

# Compute the avarage silhouette for each n_clusters in range 2-50
def clusterize(X):
    sil = dict()
    for k in range(2,51):
       clustering = AgglomerativeClustering(n_clusters=k, compute_full_tree=True).fit(X)
       sil[k] = silhouette_score(X, clustering.labels_)
    return sil

# The first AgglomerativeClustering is used to create the dendogram since sklearn requires a distance_threshold to return
#  the distances between clusters, the second time we actually compute the best cluster based on silhouettes 
# couldn't find a way to save the clustering steps so we recompute the best cluster
# X must be already shaped as a matrix of shape (n_samples, n_features)
def get_best_cluster(X, sil):
    maximum = max(sil, key=sil.get)
    print(sil[maximum])
    dendo_clustering = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=0).fit(X)
    plot_dendrogram(dendo_clustering, maximum)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    model = AgglomerativeClustering(n_clusters=maximum, compute_full_tree=True).fit(X)
    return model

def reshape_matrix(X, n_snaps):
    return X.reshape((n_snaps, -1))

def plot_dendrogram(model, n_cluster, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    
    # To show the right cut on the dendogram we search for the treshold of the best cluster in the linkage matrix
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, color_threshold=linkage_matrix[len(linkage_matrix)-n_cluster][2], **kwargs) 

""" def hamming_cluster(n_cluster, X, n_sanps, out_dir="fig/"):
    c_type = "hamming_distance"
    X = X.reshape((n_sanps,-1))
    Z = linkage(X, metric='hamming')
    clustering = fcluster(Z,t=5 , criterion='maxclust')
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(Z)
    plt.savefig(out_dir + "dendrogram_" + c_type + ".png")
    plt.show()
    plt.close()
    return clustering """

"""def plot_dendrogram_old(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    print(linkage_matrix)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)"""