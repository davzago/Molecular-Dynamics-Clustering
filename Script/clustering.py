from sklearn.cluster import AgglomerativeClustering
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def euclidean_cluster(n_cluster, X, out_dir="../fig/"):
    c_type = "euclidean_distance"
    clustering = AgglomerativeClustering(n_clusters=n_cluster, compute_full_tree=True).fit(X)
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(clustering, truncate_mode='level', p=3)
    plt.savefig(out_dir + "dendrogram_" + c_type + ".png")
    plt.show()
    plt.close()
    return clustering


def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)