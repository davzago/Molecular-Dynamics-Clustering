import Snapshot
import os
import re

def parse(data_path): 
    with open(data_path) as f:
        lines = [line.rstrip() for line in f]

    snap = []
    for l in lines:
        print(l)
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
            residue1 = (int(split_1[0].split(":")[1]), split_1[0].split(":")[0])
            residue2 = (int(split_1[2].split(":")[1]), split_1[0].split(":")[0])
            snapshot[n].addNode(residue1)
            snapshot[n].addNode(residue2)
            snapshot[n].addEdge(residue1, residue2, interaction, distance, energy)
        snapshot[n].nodes.sort()
        n += 1
    return snapshot



def parse2(data_path): 
    with open(data_path) as f:
        path = f.readline()
    snap = []
    directory = os.fsencode(path)
    files = sorted(os.listdir(directory), key=lambda s : int("".join(re.findall(r'\d+', s.decode('utf-8'))))) # kind of ugly but useful to keep the files in the right order (not needed if using parse instead of parse2)
    for file in files:
        filename = os.fsdecode(file)
        print(filename)
        f = open(path + "/" + filename, 'r', encoding='ISO-8859-1')
        next(f)
        snap.append([line.rstrip() for line in f])

    snapshot = []
    n = 0
    for s in snap:
        snapshot.append(Snapshot.Snapshot(n))
        for data in s:
            split_1 = data.split("\t")
            interaction = split_1[1]
            distance = float(split_1[3])
            energy = float(split_1[5])
            residue1 = (int(split_1[0].split(":")[1]), split_1[0].split(":")[0])
            residue2 = (int(split_1[2].split(":")[1]), split_1[0].split(":")[0])
            snapshot[n].addNode(residue1)
            snapshot[n].addNode(residue2)
            snapshot[n].addEdge(residue1, residue2, interaction, distance, energy)
        snapshot[n].nodes.sort()
        n += 1
    return snapshot