class Snapshot:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.simple_edges = []
        self.type_edges = []

    def addNode(self, node):
        if(node not in self.nodes):
            self.nodes.append(node)

    def addEdge(self, node1, node2, interaction_type, distance, energy):
        self.edges.append(tuple([node1, node2, interaction_type, distance, energy]))

    def addSimpleEdge(self, node1, node2):
        self.simple_edges.append(tuple([node1, node2]))

    def addTypeEdge(self, node1, node2, interaction_type, energy):
        self.type_edges.append(tuple([node1, node2, interaction_type, energy]))