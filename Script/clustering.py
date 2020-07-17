from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score , normalized_mutual_info_score
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from yellowbrick.cluster import KElbowVisualizer
import warnings

def reshape_matrix(X, n_snaps):
    return X.reshape((n_snaps, -1))

# Plots the dendogram of the obtained clustering also adding the cut line
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
    dendrogram(linkage_matrix, color_threshold=linkage_matrix[len(linkage_matrix)-n_cluster][2]+0.00001, **kwargs) 
    cut = (linkage_matrix[len(linkage_matrix)-n_cluster][2] + linkage_matrix[len(linkage_matrix)-n_cluster+1][2]) / 2
    plt.hlines(cut,0, 10000, color='r', linewidth=0.5)

# applies the PCA trasformation on the scaled data, obtaining a matrix with less n_features (n_features is the number of features necessary to explain 90% of the variance of the data)
def PCA_transform(X):
    return PCA(n_components=0.9).fit_transform(X)

# method which tries every possible clustering from(4 to 50) to the output the best number of cluster
def elbow(X, path):
    model = AgglomerativeClustering(affinity='cosine', linkage='average') # affinity='euclidean', linkage='ward'
    visualizer = KElbowVisualizer(model, k=(4,50), metric='silhouette', timings=False)
    visualizer.fit(X)
    dendo_clustering = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=0,
                                                 linkage='average', affinity='cosine').fit(X)
    plt.figure(figsize=(20,10), dpi=200)
    plot_dendrogram(dendo_clustering, visualizer.elbow_value_)
    plt.xlabel("Number of the snapshot")
    plt.ylabel("cosine distance")
    plt.grid(b=None)
    plt.savefig(path + "/dendogram.png")
    plt.clf()
    plt.close()
    return AgglomerativeClustering(n_clusters=visualizer.elbow_value_, affinity='cosine', linkage='average').fit(X), visualizer.elbow_value_

# Like elbow but for the RMSD clustering
def elbow_RMSD(X):
    model = AgglomerativeClustering(compute_full_tree=True, affinity='precomputed', linkage='average')
    visualizer = KElbowVisualizer(model, k=(4,50), metric='silhouette', timings=False)
    visualizer.fit(X)
    plt.clf()
    plt.close()
    dendo_clustering = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=0, affinity='precomputed', linkage='average').fit(X)
    plot_dendrogram(dendo_clustering, visualizer.elbow_value_)
    plt.grid(b=None)
    #plt.show()
    plt.close()
    return AgglomerativeClustering(n_clusters=visualizer.elbow_value_, affinity='precomputed', linkage='average').fit(X)

# Returns the list of contact maps for each cluster
def clusterize_snaps(best_k, labels, matrix_snap):
    clusterized_snaps = []
    for k in range(0,best_k):
        k_list = []
        for i in range(0,len(labels)):
            if labels[i]==k:
                for m in range(0,len(matrix_snap[i])):
                    for n in range(0,len(matrix_snap[i])):
                        if matrix_snap[i,m,n] > 0:
                            matrix_snap[i,m,n] = 1
                k_list.append(matrix_snap[i])
        clusterized_snaps.append(k_list)
    return clusterized_snaps

# Computes the common contacts 
def get_common_contacts(clusterized_snaps):
    common_snaps = []
    for k in range(0,len(clusterized_snaps)):
        n_snaps = len(clusterized_snaps[k])
        n, _ = clusterized_snaps[k][0].shape
        common_contacts = np.zeros((n,n))
        for i in  range(0,n_snaps):
            for j in range(0,n):
                for w in range(0,n):
                    common_contacts[j,w] += clusterized_snaps[k][i][j,w]
        for j in range(0,n):
            for w in range(0,n):
                if common_contacts[j,w] < n_snaps/2:
                    common_contacts[j,w] = 0
                else:
                    common_contacts[j,w] = 1
        common_snaps.append(common_contacts)
    return common_snaps

# Computes the list of important contacts for each cluster
def get_important_contacts(common_contacts):
    important_contacts = []
    k = len(common_contacts)
    n, _ = common_contacts[0].shape
    for x in range(0,k):
        for i in range(0,n):
                for j in range(0,n):
                    w = 0
                    for y in range(0,k):
                        if x != y and common_contacts[x][i,j] == 1 and common_contacts[y][i,j] == 0:
                            w += 1
                    if w == k-1:
                        important_contacts.append((i,j,x))
    return important_contacts