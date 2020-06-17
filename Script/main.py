import argparse
import numpy as np
from skbio.stats.distance import DissimilarityMatrix
from matplotlib import pyplot as plt

import parsing

parser = argparse.ArgumentParser(description='Hierarchical clustering using RING data.')
parser.add_argument('data_path', help='File with the path to RING contact map files (edge files)')
parser.add_argument('-conf', help='Configuration file with algorithm parameters')
parser.add_argument('-out_dir', help='Output directory')
parser.add_argument('-tmp_dir', help='Temporary file directory')
args = parser.parse_args()
snapSet = parsing.parse(args.data_path)


# compute the n-th snapshot's distance matrix
def calcMatrix(snap):
    n = len(snapSet[snap].nodes)
    matrix = np.zeros((n, n))
    for i in snapSet[snap].edges:
        node1 = i[0]
        node2 = i[1]
        matrix[posList[snap][node1]][posList[snap][node2]] = i[3]
    return matrix

# plot the n-th snapshot's distance matrix
# make sure that ../fig directory already exists
def plotDistanceMatrix(matrix_snap, snap):
    dmatrix = DissimilarityMatrix(matrix_snap, snapSet[snap].nodes)
    fig = dmatrix.plot(cmap='Reds')
    plt.savefig("../fig/" + str(snap) + ".png")
    plt.close()


# dict per memorizzare, per ogni snapshot, per ogni nodo, la posizione nella matrice
posList = [dict() for x in range(0, len(snapSet)-1)]
for snap in range (0, len(snapSet)-1):
    pos = 0
    for node in snapSet[snap].nodes:
        posList[snap][node] = pos
        pos += 1

for i in range(0, len(snapSet)-1):
    matrix_snap = calcMatrix(i)
    # plotDistanceMatrix(matrix_snap, i)

# np.savetxt("../foo.csv", matrix, delimiter=",")

""" 
### DISTINCT EDGES AND OCCURENCE COUNT IN ALL SNAPSHOT
# key dict sono edges, value sono occorrenze dell'edge key
count = dict()
for i in range(0, len(snapSet)-1):
    for s in snapSet[i].edges:
        if((str(s[0]) + "-" + str(s[1])) in count):
            count[str(s[0]) + "-" + str(s[1])] += 1
        else:
            count[str(s[0]) + "-" + str(s[1])] = 1

for i in count:
    print(i, ": ", count[i])

print(len(count))
 """