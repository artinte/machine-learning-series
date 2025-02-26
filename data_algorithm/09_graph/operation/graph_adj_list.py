class Vertex:
    def __init__(self, val):
        self.val = val


def vals_to_sets(vals):
    return [Vertex(val) for val in vals]


class GraphAdjList:
    def __init__(self, edges: list[list[Vertex]]):
        # 邻接表，key: 顶点，value: 该顶点的所有邻接顶点
        self.adj_list = dict[Vertex, list[Vertex]]()
        # 添加所有顶点和边
        for edge in edges:
            self.add_vertex(edge[0])
            self.add_vertex(edge[1])
            self.add_edge(edge[0], edge[1])

    def size(self):
        return len(self.adj_list)

    def add_edge(self, vet1, vet2):
        if vet1 not in self.adj_list or vet2 not in self.adj_list or vet1 == vet2:
            raise ValueError()
        # 添加边 vet1 - vet2
        self.adj_list[vet1].append(vet2)
        self.adj_list[vet2].append(vet1)

    def remove_edge(self, vet1, vet2):
        if vet1 not in self.adj_list or vet2 not in self.adj_list or vet1 == vet2:
            raise ValueError()

        # 删除边 vet1 - vet2
        self.adj_list[vet1].remove(vet2)
        self.adj_list[vet2].remove(vet1)

    def add_vertex(self, vet):
        if vet in self.adj_list:
            return
        self.adj_list[vet] = []

    def remove_vertex(self, vet):
        if vet not in self.adj_list:
            raise ValueError()
        # 在邻接表中删除顶点 vet 对应的链表
        self.adj_list.pop(vet)
        # 遍历其它顶点的链表，删除所有包含 vet 的边
        for vertex in self.adj_list:
            if vet in self.adj_list[vertex]:
                self.adj_list[vertex].remove(vet)

    def print(self):
        for vertex in self.adj_list:
            tmp = [v.val for v in self.adj_list[vertex]]
            print(f"{vertex.val}: {tmp}")


v = vals_to_sets([1, 3, 2, 5, 4])
edges = [
    [v[0], v[1]],
    [v[0], v[3]],
    [v[1], v[2]],
    [v[2], v[3]],
    [v[2], v[4]],
    [v[3], v[4]],
]

graph = GraphAdjList(edges)

# 添加边
graph.add_edge(v[0], v[2])
# 删除边
graph.remove_edge(v[0], v[1])
# 添加顶点
v5 = Vertex(6)
graph.add_vertex(v5)
# 删除顶点
graph.remove_vertex(v[1])
