from sklearn.cluster import AgglomerativeClustering
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def euclidean_cluster(n_cluster, X, out_dir="../fig/"):
    c_type = "euclidean_distance"
    clustering = AgglomerativeClustering(n_clusters=n_cluster, compute_full_tree=True),fit(X)
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(clustering, truncate_mode='level', p=3)
    plt.savefig(out_dir + "dendrogram_" + c_type + ".png")
    plt.show()
    plt.close()


def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)