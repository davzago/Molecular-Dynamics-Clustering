import argparse
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

import parsing
import matrix
import clustering
import RMSD

parser = argparse.ArgumentParser(description='Hierarchical clustering using RING data.')
parser.add_argument('data_path', help='File with the path to RING contact map files (edge files)')
parser.add_argument('pdb_path', help='Path to the pdb directory')
parser.add_argument('-conf', help='Configuration file with algorithm parameters')
parser.add_argument('-out_dir', help='Output directory')
parser.add_argument('-tmp_dir', help='Temporary file directory')
args = parser.parse_args()

pdb_path = args.pdb_path 
snapSet = parsing.parse2(args.data_path)
n_sanps = len(snapSet)


# node_list contiene tutti i nodi che compaiono almeno una volta in uno snapshot
node_list = []
for i in range(0, n_sanps):
    for s in snapSet[i].nodes:
        if(s not in node_list):
            node_list.append(s)
node_list.sort()

# dict per memorizzare, per ogni snapshot, per ogni nodo, la posizione nella matrice
# si potrebbe spostare come proprietà di Snapshot, così da avere un dict distinto per ogni snapshot
node_position = dict()
pos = 0
for node in node_list:
    node_position[node] = pos
    pos += 1

n_nodes = len(node_list)
matrix_snap = np.zeros((n_sanps,n_nodes,n_nodes))
for i in range(0, n_sanps):
    matrix_snap[i] = matrix.calcMatrix(i, snapSet, node_position)
    # matrix.plotDistanceMatrix(matrix_snap, i, snapSet)

#labels = clustering.euclidean_cluster(4, matrix_snap, n_sanps, "../dendrogram/", 80)
#print(labels)
#print(n_sanps,labels.shape)
""" RMSD_array = RMSD.get_dense_array(pdb_path)
Z = linkage(RMSD_array, method='ward')
dendrogram(Z, 0)
plt.show()
plt.close() """
    

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

sil = clustering.clusterize(matrix_snap, n_sanps)

model = clustering.get_best_cluster(matrix_snap, n_sanps, sil)


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

