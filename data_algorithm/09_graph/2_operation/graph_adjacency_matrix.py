class GraphAdjMatrix:
    """
    基于邻接矩阵实现的无向图类
    """

    def __init__(self, vertices: list[int], edges: list[list[int]]):
        self.vertices = []
        self.adj_mat = []
        # 添加顶点
        for val in vertices:
            self.add_vertex(val)
        # 添加边
        for e in edges:
            self.add_edge(e[0], e[1])

    def size(self):
        return len(self.vertices)

    def add_vertex(self, val):
        n = self.size()
        # 向顶点列表中添加新顶点的值
        self.vertices.append(val)
        # 在邻接矩阵中添加一行
        new_row = [0] * n
        self.adj_mat.append(new_row)
        # 在邻接矩阵中添加一列
        for row in self.adj_mat:
            row.append(0)

    def remove_vertex(self, index):
        if index >= self.size():
            raise IndexError()
        # 在顶点列表中移除索引 index 的顶点
        self.vertices.pop(index)
        # 在邻接矩阵中删除索引 index 的行
        self.adj_mat.pop(index)
        # 在邻接矩阵中删除索引 index 的列
        for row in self.adj_mat:
            row.pop(index)

    def add_edge(self, i, j):
        if i < 0 or j < 0 or i >= self.size() or j >= self.size() or i == j:
            raise IndexError()
        # 在无向图中，邻接矩阵关于主对角线对称，即满足 (i, j) == (j, i)
        self.adj_mat[i][j] = 1
        self.adj_mat[j][i] = 1

    def remove_edge(self, i, j):
        if i < 0 or j < 0 or i >= self.size() or j >= self.size() or i == j:
            raise IndexError()
        self.adj_mat[i][j] = 0
        self.adj_mat[j][i] = 0

    def print(self):
        print("顶点列表 =", self.vertices)


vertices = [1, 3, 2, 5, 4]
edges = [[0, 1], [0, 3], [1, 2], [2, 3], [2, 4], [3, 4]]
graph = GraphAdjMatrix(vertices, edges)
# 顶点 1，2 的索引分别为 0，2
graph.add_edge(0, 2)
graph.remove_edge(0, 1)
graph.add_vertex(6)
graph.remove_vertex(1)
