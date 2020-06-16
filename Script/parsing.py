import argparse
import Snapshot

### $ python3 stru.py contact_maps_paths.txt -conf params.ini -out_dir results/ -tmp_dir /tmp

parser = argparse.ArgumentParser(description='Hierarchical clustering using RING data.')
parser.add_argument('data_path', help='File with the path to RING contact map files (edge files)')
parser.add_argument('-conf', help='Configuration file with algorithm parameters')
parser.add_argument('-out_dir', help='Output directory')
parser.add_argument('-tmp_dir', help='Temporary file directory')
args = parser.parse_args()

with open(args.data_path) as f:
    lines = [line.rstrip() for line in f]

snap = []
for l in lines:
    with open(l) as s:
        next(s)
        snap.append([line.rstrip() for line in s])

snapshot = []
n = 0
for s in snap:
    snapshot.append(Snapshot.Snapshot(n))
    for data in s:
        split_1 = data.split("\t")
        interaction = split_1[1]
        distance = float(split_1[3])
        energy = float(split_1[5])
        atom1 = int((split_1[0].split(":"))[1])
        atom2 = int((split_1[2].split(":"))[1])
        snapshot[n].addNode(atom1)
        snapshot[n].addNode(atom2)
        snapshot[n].addEdge(atom1, atom2, interaction, distance, energy)
    snapshot[n].nodes.sort()
    n += 1

