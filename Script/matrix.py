import numpy as np
from matplotlib import pyplot as plt
import math

# compute the n-th snapshot's distance matrix
def calcMatrix(snap, snapSet, posList):
    n = len(posList)
    matrix = np.zeros((n, n))
    for i in snapSet[snap].edges:
        node1 = i[0]
        node2 = i[1]
        if(posList[node1] > posList[node2]): 
            tmp = node1
            node1 = node2
            node2 = tmp
        if(matrix[posList[node1]][posList[node2]] == 0):
            matrix[posList[node1]][posList[node2]] = i[4] + 0.2 * abs(posList[node1] - posList[node2])
        else:
            matrix[posList[node1]][posList[node2]] += i[4] + 0.2 * abs(posList[node1] - posList[node2])
    return matrix

def calcVector(snap, snapSet, vector_position):
    n = len(vector_position)
    vector_edges = np.zeros(n)
    for i in snapSet[snap].type_edges:
        # if((abs(i[0][0] - i[1][0]) > 10) and (i[0][1] == i[1][1])):   # condizione per escludere contatti fra residui "vicini" nella sequenza
        vector_edges[vector_position[i]] = i[3] + 0.2 * abs(i[0][0] - i[1][0])
    return vector_edges

def calcVectorSimple(snap, snapSet, vector_position_simple):
    n = len(vector_position_simple)
    vector_edges_simple = np.zeros(n)
    for i in snapSet[snap].type_edges:
        i_simple = tuple([i[0], i[1]])
        # if((abs(i[0][0] - i[1][0]) > 10) and (i[0][1] == i[1][1])):   # condizione per escludere contatti fra residui "vicini" nella sequenza
        if(vector_edges_simple[vector_position_simple[i_simple]] == 0):
            vector_edges_simple[vector_position_simple[i_simple]] = i[3] + 0.1 * abs(i[0][0] - i[1][0])
            # print(i_simple, " pos:", vector_position_simple[i_simple]," --- New --- ", vector_edges_simple[vector_position_simple[i_simple]])
        else:
            vector_edges_simple[vector_position_simple[i_simple]] += i[3] + 0.1 * abs(i[0][0] - i[1][0])
            # print(i_simple, " pos:", vector_position_simple[i_simple]," --- Old --- ", vector_edges_simple[vector_position_simple[i_simple]])
    return vector_edges_simple

# plot the n-th snapshot's distance matrix
# make sure that ../fig directory already exists
# PS: SI CONSIDERANO SOLAMENTE I NODI CHE SONO PRESENTI NELL'MD, QUINDI NON MOLTO INDICATIVO VISIVAMENTE
""" def plotDistanceMatrix(matrix_snap, snap):
    plt.imshow(matrix_snap[snap], cmap='Reds', interpolation='nearest')#, vmin=0, vmax=30)
    plt.colorbar()
    plt.savefig("../fig/" + str(snap) + ".png")
    plt.close() """

def ignore_diagonal(matrix_snap):
    no_diagonal_snaps = []
    for k in range(0,matrix_snap.shape[0]):
        contacts = []
        for i in range(0, matrix_snap.shape[1]-100):
            for j in range(i+100, matrix_snap.shape[1]):
                contacts.append(matrix_snap[k,i,j])
        no_diagonal_snaps.append(contacts)
    return np.array(no_diagonal_snaps)

def condense_matrix(snapshot, kernel_dim):
    if snapshot.shape[0] % kernel_dim == 0:
        new_dim = int(snapshot.shape[0] / kernel_dim)
        dense_matrix = np.zeros((new_dim,new_dim))
        for i in range(0,new_dim):
            for j in range(0,new_dim):
                for sub_i in range(0,kernel_dim):
                    for sub_j in range(0,kernel_dim):  
                        dense_matrix[i,j] += snapshot[i*kernel_dim+sub_i, j*kernel_dim+sub_j]
        return dense_matrix
    else: return np.zeros((3,3)) 

def edge_count(snapSet):
    count = dict()
    for i in range(0, len(snapSet)):
        for s in snapSet[i].edges:
            if((str(s[0]) + "-" + str(s[1]) + "-" + str(s[2])) in count):
                count[str(s[0]) + "-" + str(s[1]) + "-" + str(s[2])] += 1
            else:
                count[str(s[0]) + "-" + str(s[1]) + "-" + str(s[2])] = 1
    return sorted(count.items(), key=lambda x: x[1], reverse=False)
    

def output_edge_count(count_sorted, path):
    edges_count = open(path + "/edges_count.txt","w")
    for i in count_sorted:
        edges_count.write(str(i[0]) + ": " + str(i[1]) + "\n")
    edges_count.close()

def output_distance_matrix(distance_matrix, path):
    n = distance_matrix.shape[0]
    resultFile = open(path + "/distance_matrix.txt","w")
    for i in range(0,n):
        for j in range(0,n):
            resultFile.write(str(distance_matrix[i,j]) + " ")
        resultFile.write("\n")
    resultFile.close()

def output_labels(labels, path):
    resultFile = open(path + "/labels.txt","w")
    for i in range(0,len(labels)):
        resultFile.write("snap " + str(i+1) + ": " + str(labels[i]) + "\n")
    resultFile.close()

def output_common_contacts(common_contacts, path):
    for i in range(0,len(common_contacts)):
        plt.figure(figsize=(10,10))
        plt.imshow(common_contacts[i], cmap='Reds', interpolation='nearest')
        plt.colorbar()
        plt.savefig(path + "/contact_map_" + str(i) + ".png", dpi=100)
        plt.close()

def output_important_contacts(important_list, node_list, path):
    resultFile = open(path + "/important_contacts.txt","w")
    for i in range(0,len(important_list)):
        node1 = node_list[important_list[i][0]]
        node2 = node_list[important_list[i][1]]
        cluster = important_list[i][2]
        resultFile.write(str(node1) + " - " + str(node2) + " in cluster " + str(cluster) + "\n")
    resultFile.close()