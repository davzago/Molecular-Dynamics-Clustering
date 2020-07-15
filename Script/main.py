import argparse
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import operator
import sys
import warnings
import os

import parsing
import matrix
import clustering
import RMSD
import RMSD_out


parser = argparse.ArgumentParser(description='Hierarchical clustering using RING data.')
parser.add_argument('data_path', help='File with the path to RING contact map files (edge files)')
parser.add_argument('-RMSD_path', help='Path to the file containing the distance matrix calculated with the TM-Score script (if not calculated use -path_to_pdb)')
parser.add_argument('-path_to_pdb', help='The path to the pdb folder that is used to calculate the RMSD (to use if the RMSD has not already been calculated) the result will be put in a file named RMSD.txt in the folder named like the data_path in out_dir, also to use this command the TMscore.cpp is necessary')
parser.add_argument('-conf', help='Configuration file with algorithm parameters')
parser.add_argument('-out_dir', help='Output directory', default='../out_dir')
parser.add_argument('-tmp_dir', help='Temporary file directory', default='../tmp_dir')
args = parser.parse_args()

snapSet = parsing.parse(args.data_path)
n_snaps = len(snapSet)

# define the name of the directory to be created
splits = args.data_path.split('/')
path = "../out_dir/" + splits[len(splits)-1].split('.')[0]

try:
    os.mkdir(path)
except OSError:
    print ("%s Direcotry already exists, the already existing direcory will be used" % path)
else:
    print ("Successfully created the directory %s " % path)


# node_list contiene tutti i nodi che compaiono almeno una volta in uno snapshot
node_list = []
for i in range(0, n_snaps):
    for s in snapSet[i].nodes:
        if(s not in node_list):
            node_list.append(s)
node_list.sort(key = operator.itemgetter(1, 0))

# dict per memorizzare, per ogni snapshot, per ogni nodo, la posizione nella matrice
# si potrebbe spostare come proprietà di Snapshot, così da avere un dict distinto per ogni snapshot
node_position = dict()
pos = 0
for node in node_list:
    node_position[node] = pos
    pos += 1


n_nodes = len(node_list)
matrix_snap = np.zeros((n_snaps,n_nodes,n_nodes))
for i in range(0, n_snaps):
    matrix_snap[i] = matrix.calcMatrix(i, snapSet, node_position)
    #matrix.plotDistanceMatrix(matrix_snap, i, snapSet)

pos = 0
vector_position = dict()
contact_list = []
for i in range(0, n_snaps):
    for s in snapSet[i].type_edges:
        if(s not in vector_position):
            contact_list.append(s)
            vector_position[s] = pos
            pos += 1
n_distinct_edges = len(contact_list)
vector_edges = np.zeros((n_snaps, n_distinct_edges))
for i in range(0, n_snaps):
    vector_edges[i] = matrix.calcVector(i, snapSet, vector_position)

pos = 0
vector_position_simple = dict()
contact_list_simple = []
for i in range(0, n_snaps):
    for s in snapSet[i].simple_edges:
        if(s not in vector_position_simple):
            contact_list_simple.append(s)
            vector_position_simple[s] = pos
            pos += 1
n_distinct_edges_simple = len(contact_list_simple)
vector_edges_simple = np.zeros((n_snaps, n_distinct_edges_simple))
for i in range(0, n_snaps):
    vector_edges_simple[i] = matrix.calcVectorSimple(i, snapSet, vector_position_simple)

scaled_vector = clustering.StandardScaler().fit_transform(vector_edges_simple)


np.set_printoptions(threshold=sys.maxsize)
no_diag_snaps = matrix.ignore_diagonal(matrix_snap)





    

### DISTINCT EDGES AND OCCURENCE COUNT IN ALL SNAPSHOT
# key dict sono edges, value sono occorrenze dell'edge key (valuta residue1, residue2, interaction)
count = dict()
for i in range(0, len(snapSet)):
    for s in snapSet[i].edges:
        if((str(s[0]) + "-" + str(s[1]) + "-" + str(s[2])) in count):
            count[str(s[0]) + "-" + str(s[1]) + "-" + str(s[2])] += 1
        else:
            count[str(s[0]) + "-" + str(s[1]) + "-" + str(s[2])] = 1

count_sorted = sorted(count.items(), key=lambda x: x[1], reverse=False)


"""distance matrix = clustering.squareform(clustering.pdist(vector_edges, metric='yule'))
clustering.test_metric(X[0,:], RMSD_distance_matrix[0,:])"""

X = clustering.reshape_matrix(matrix_snap, n_snaps)
X_PCA = clustering.PCA_transform(scaled_vector)
#X_SVD = clustering.SVD_transform(vector_edges_simple)
distance_matrix = clustering.squareform(clustering.pdist(X_PCA, metric='cosine'))
matrix.output_distance_matrix(distance_matrix, path)


model, best_k = clustering.elbow(X_PCA, path)

if args.path_to_pdb is not None:
    RMSD_out.get_distance_matrix(args.path_to_pdb, path)



if args.RMSD_path is not None:
    RMSD_distance_matrix = RMSD.get_distance_matrix_from_file(args.RMSD_path)
    model_RMSD = clustering.elbow_RMSD(RMSD_distance_matrix)
    rand_index = clustering.adjusted_rand_score(model_RMSD.labels_,model.labels_)
    mutal_info_score = clustering.normalized_mutual_info_score(model_RMSD.labels_,model.labels_)
    #print("mutual info score between RMSD clustering and contact clustering:", mutal_info_score)
    print("RandIndex between RMSD clustering and contact clustering:", rand_index)

clusterized_snaps = clustering.clusterize_snaps(best_k, model.labels_, matrix_snap)
common_contacts = clustering.get_common_contacts(clusterized_snaps)
important_list = clustering.get_important_contacts(common_contacts)

matrix.output_labels(model.labels_, path)
matrix.output_common_contacts(common_contacts, path)
matrix.output_imprtant_contacts(important_list, node_list, path)

""" edges_count = open("edges_count.txt","w")
for i in count_sorted:
    edges_count.write(str(i[0]) + ": " + str(i[1]) + "\n")
    # print(i[0], ": ", i[1])
edges_count.close() """

""" ### test hamming distance. Scartato, snap hanno un rapporto contatti_diversi/contatti_totali troppo simili tra loro 
### vector that contain the presence-bit for every edges in all snapshot
pos = 0
vector_position = dict()
contact_list = []
for i in range(0, n_sanps):
    for s in snapSet[i].simple_edges:
        if(s not in vector_position):
            contact_list.append(s)
            vector_position[s] = pos
            pos += 1
n_distinct_edges = len(contact_list)
contact_vector = np.full((n_sanps,n_distinct_edges), 0)
for i in range(0, n_sanps):
    for e in range(0, len(contact_list)):
            if(contact_list[e] in snapSet[i].simple_edges):
                contact_vector[i][e] = 1

distance = []
n = 0
for i in range(0, n_sanps-1):
    for e in range(i+1, n_sanps):
        same_value = np.count_nonzero(np.logical_and(contact_vector[i] == contact_vector[e], contact_vector[i] != 0))
        different_value = np.count_nonzero(contact_vector[i] != contact_vector[e])
        print(same_value, " /// ", different_value)
        distance.append(different_value/(same_value+different_value))
        print("(", i, " - ", e, ") --> ", distance[n])
        n += 1
        # print(i, " : ", e, " = ", distance.hamming(contact_vector[i], contact_vector[e]))
labels = clustering.hamming_cluster(4, contact_vector, n_sanps, "../dendrogram/")
print(labels) """

