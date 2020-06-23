import Snapshot

def parse(data_path): 
    with open(data_path) as f:
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
            # Ci basta l'intero o ci serve tutta la stringa? penso che i numeri si ripetano in uno snapshot indicando però nodi diversi
            atom1 = int((split_1[0].split(":"))[1])
            atom2 = int((split_1[2].split(":"))[1])
            snapshot[n].addNode(atom1)
            snapshot[n].addNode(atom2)
            snapshot[n].addEdge(atom1, atom2, interaction, distance, energy)
        snapshot[n].nodes.sort()
        n += 1
    return snapshot



