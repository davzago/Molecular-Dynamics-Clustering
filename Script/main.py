import argparse

import parsing
import matrix

parser = argparse.ArgumentParser(description='Hierarchical clustering using RING data.')
parser.add_argument('data_path', help='File with the path to RING contact map files (edge files)')
parser.add_argument('-conf', help='Configuration file with algorithm parameters')
parser.add_argument('-out_dir', help='Output directory')
parser.add_argument('-tmp_dir', help='Temporary file directory')
args = parser.parse_args()

snapSet = parsing.parse(args.data_path)

# nodeList contiene tutti i nodi che compaiono almeno una volta in uno snapshot
nodeList = []
for i in range(0, len(snapSet)-1):
    for s in snapSet[i].nodes:
        if(s not in nodeList):
            nodeList.append(s)

nodeList.sort()

# dict per memorizzare, per ogni snapshot, per ogni nodo, la posizione nella matrice
# si potrebbe spostare come proprietà di Snapshot, così da avere un dict distinto per ogni snapshot
posList = dict()
pos = 0
for node in nodeList:
    posList[node] = pos
    pos += 1

for i in range(0, len(snapSet)):
    matrix_snap = matrix.calcMatrix(i, snapSet, posList)
    # matrix.plotDistanceMatrix(matrix_snap, i, snapSet)
    


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