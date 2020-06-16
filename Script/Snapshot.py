class Snapshot:
    def __init__(self, id):
        self.id = id
        self.nodes = []
        self.edges = []

    def addNode(self, node):
        if(node not in self.nodes):
            self.nodes.append(node)

    def addEdge(self, node1, node2, interaction_type, distance, energy):
        self.edges.append([node1, node2, interaction_type, distance, energy])