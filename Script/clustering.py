from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score , normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA, TruncatedSVD, NMF
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from yellowbrick.cluster import KElbowVisualizer

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
    for k in range(4,51):
       clustering = AgglomerativeClustering(n_clusters=k, compute_full_tree=True, linkage='average', affinity='cosine').fit(X)
       sil[k] = silhouette_score(X, clustering.labels_, metric='cosine')
    return sil

# The first AgglomerativeClustering is used to create the dendogram since sklearn requires a distance_threshold to return
#  the distances between clusters, the second time we actually compute the best cluster based on silhouettes 
# couldn't find a way to save the clustering steps so we recompute the best cluster
# X must be already shaped as a matrix of shape (n_samples, n_features)
def get_best_cluster(X, sil):
    maximum = max(sil, key=sil.get)
    print("maximum contacts silhouette:" ,sil[maximum], "with", maximum, "clusters")
    dendo_clustering = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=0,
                                                 linkage='average', affinity='cosine').fit(X)
    plot_dendrogram(dendo_clustering, maximum)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    model = AgglomerativeClustering(n_clusters=maximum, compute_full_tree=True, linkage='average', affinity='cosine').fit(X)
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
    dendrogram(linkage_matrix, color_threshold=linkage_matrix[len(linkage_matrix)-n_cluster][2]+0.00001, **kwargs) 

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

def clusterize_RMSD(X):
    sil = dict()
    for k in range(4,51):
       clustering = AgglomerativeClustering(n_clusters=k, compute_full_tree=True, affinity='precomputed', linkage='average').fit(X)
       sil[k] = silhouette_score(X, clustering.labels_)
    return sil

def get_best_cluster_RMSD(X, sil):
    maximum = max(sil, key=sil.get)
    print("maximum RMSD silhouette:", sil[maximum], "with", maximum, "clusters")
    dendo_clustering = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=0, affinity='precomputed', linkage='average').fit(X)
    plot_dendrogram(dendo_clustering, maximum)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    plt.clf()
    plt.close()
    model = AgglomerativeClustering(n_clusters=maximum, compute_full_tree=True, affinity='precomputed', linkage='average').fit(X)
    return model

def get_randIndex(contacts_labels, RMSD_labels):
    return adjusted_rand_score(contacts_labels, RMSD_labels)

def PCA_transform(X):
    print("PCA")
    return PCA(n_components=0.9).fit_transform(X)

def SVD_transform(X):
    print("SVD")
    svd = TruncatedSVD(n_components=75, random_state=42)
    X_transformed = svd.fit_transform(X)
    print("explained variance:", sum(svd.explained_variance_ratio_))
    return X_transformed

def NMF_transform(X):
    print("NMF")
    nmf = NMF(n_components=75, init='random', random_state=42, max_iter=20)
    return nmf.fit_transform(X)

def pearson_affinity(M):
    return 1 - np.array([[pearsonr(a,b)[0] for a in M] for b in M])

def pearson_metric(x, y):
    return 1 - pearsonr(x,y)[0]

def test_metric(x ,y):
    x_dict = dict()
    y_dict = dict()
    
    for i in range(0,len(x)):
        x_dict[i] = x[i]
    for i in range(0,len(y)):
        y_dict[i] = y[i]
    x_sorted = sorted(x_dict.items(), key = lambda x: x[1])
    y_sorted = sorted(y_dict.items(), key = lambda x: x[1])
    print([x[0] for x in x_sorted])
    print([y[0] for y in y_sorted])

def elbow(X):
    model = AgglomerativeClustering(affinity='cosine', linkage='average') # affinity='euclidean', linkage='ward'
    visualizer = KElbowVisualizer(model, k=(4,50), metric='silhouette', timings=False)
    visualizer.fit(X)
    visualizer.show()
    plt.clf()
    plt.close()
    dendo_clustering = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=0,
                                                 linkage='average', affinity='cosine').fit(X)
    plot_dendrogram(dendo_clustering, visualizer.elbow_value_)
    plt.grid(b=None)
    plt.show()
    plt.clf()
    plt.close()
    return AgglomerativeClustering(n_clusters=visualizer.elbow_value_, affinity='cosine', linkage='average').fit(X), visualizer.elbow_value_

def elbow_RMSD(X):
    model = AgglomerativeClustering(compute_full_tree=True, affinity='precomputed', linkage='average')
    visualizer = KElbowVisualizer(model, k=(4,50), metric='silhouette', timings=False)
    visualizer.fit(X)
    visualizer.show()
    plt.clf()
    plt.close()
    dendo_clustering = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=0, affinity='precomputed', linkage='average').fit(X)
    plot_dendrogram(dendo_clustering, visualizer.elbow_value_)
    plt.grid(b=None)
    plt.show()
    plt.close()
    return AgglomerativeClustering(n_clusters=visualizer.elbow_value_, affinity='precomputed', linkage='average').fit(X)

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
                if common_contacts[j,w] < n_snaps:
                    common_contacts[j,w] = 0
                else:
                    common_contacts[j,w] = 1
        common_snaps.append(common_contacts)
    return common_snaps
                    


    