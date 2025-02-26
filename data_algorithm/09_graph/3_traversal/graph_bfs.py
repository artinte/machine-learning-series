import collections
import sys

# 当前文件下执行脚本
sys.path.append("..")
from operation.graph_adj_list import Vertex, GraphAdjList, vals_to_sets


def graph_bfs(graph: GraphAdjList, start_vet: Vertex):
    """
    广度优先遍历
    使用邻接表来表示图，以便获取指定顶点的所有邻接顶点
    """
    result = []
    visited = set[Vertex]([start_vet])
    queue = collections.deque([start_vet])
    # 以 start_vet 为起点，循环直至访问完所有顶点
    while len(queue) > 0:
        vet = queue.popleft()
        result.append(vet)
        # 遍历该顶点的所有邻接顶点
        for adj_vet in graph.adj_list[vet]:
            if adj_vet in visited:
                continue
            queue.append(adj_vet)
            visited.add(adj_vet)
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
result = graph_bfs(graph, v[0])
for vet in result:
    print(vet.val)
