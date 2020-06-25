from sklearn.cluster import AgglomerativeClustering
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

def euclidean_cluster(n_cluster, X, n_snaps, out_dir="fig/", threshold=0):
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
    return clustering
    # The linkage matrix shows how the clustering is done, it has n-1 rows and 4 columns, each row shows which clusters are 
    # getting combined their distance and the numbero of observations in the new cluster, the new cluster label is the sum
    # of the labels of the 2 cluster composing it ence a cluster with a label lower than n is a cluster with a single observation


"""def plot_dendrogram(model, **kwargs):

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