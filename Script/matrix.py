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
            matrix[posList[node1]][posList[node2]] = i[4] - i[3]
        else:
            matrix[posList[node1]][posList[node2]] += i[4] - i[3]
    return matrix

# plot the n-th snapshot's distance matrix
# make sure that ../fig directory already exists
# PS: SI CONSIDERANO SOLAMENTE I NODI CHE SONO PRESENTI NELL'MD, QUINDI NON MOLTO INDICATIVO VISIVAMENTE
def plotDistanceMatrix(matrix_snap, snap, snapSet):
    print(snap)
    plt.imshow(matrix_snap[snap], cmap='Reds', interpolation='nearest')
    plt.colorbar()
    plt.savefig("../fig/" + str(snap) + ".png")
    plt.close()
