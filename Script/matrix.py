import numpy as np
from matplotlib import pyplot as plt

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
            matrix[posList[node1]][posList[node2]] = i[4]
        else:
            matrix[posList[node1]][posList[node2]] += i[4]
    return matrix

def calcVector(snap, snapSet, vector_position):
    n = len(vector_position)
    vector_edges = np.zeros(n)
    for i in snapSet[snap].type_edges:
        # if((abs(i[0][0] - i[1][0]) > 10) and (i[0][1] == i[1][1])):   # condizione per escludere contatti fra residui "vicini" nella sequenza
        vector_edges[vector_position[i]] = i[3]
    return vector_edges

# plot the n-th snapshot's distance matrix
# make sure that ../fig directory already exists
# PS: SI CONSIDERANO SOLAMENTE I NODI CHE SONO PRESENTI NELL'MD, QUINDI NON MOLTO INDICATIVO VISIVAMENTE
def plotDistanceMatrix(matrix_snap, snap, snapSet):
    print(snap)
    plt.imshow(matrix_snap[snap], cmap='Reds', interpolation='nearest')
    plt.colorbar()
    plt.savefig("../fig/" + str(snap) + ".png")
    plt.close()

def ignore_diagonal(matrix_snap):
    no_diagonal_snaps = []
    for k in range(0,matrix_snap.shape[0]):
        contacts = []
        for i in range(0, matrix_snap.shape[1]-100):
            for j in range(i+100, matrix_snap.shape[1]):
                contacts.append(matrix_snap[k,i,j])
        no_diagonal_snaps.append(contacts)
    return np.array(no_diagonal_snaps)


