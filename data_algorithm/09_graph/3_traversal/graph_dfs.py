import sys

# 当前文件下执行脚本
sys.path.append("..")
from operation.graph_adj_list import Vertex, GraphAdjList, vals_to_sets


def dfs(graph, visited, result, vet):
    result.append(vet)
    visited.add(vet)
    for adjVet in graph.adj_list[vet]:
        if adjVet in visited:
            continue
        # 递归访问邻接顶点
        dfs(graph, visited, result, adjVet)


def graph_dfs(graph: GraphAdjList, start_vet: Vertex):
    """
    深度优先遍历
    使用邻接表来表示图，以便获取指定顶点的所有邻接顶点
    """
    result = []
    visited = set[Vertex]()
    dfs(graph, visited, result, start_vet)
    return result


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
result = graph_dfs(graph, v[0])
for vet in result:
    print(vet.val)
