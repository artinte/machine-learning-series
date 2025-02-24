from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None) -> None:
        self.val = val
        self.left = left
        self.right = right


def list_to_tree_dfs(arr, i):
    if i < 0 or i >= len(arr) or arr[i] is None:
        return None
    root = TreeNode(arr[i])
    # 递归构建左右子树
    root.left = list_to_tree_dfs(arr, 2 * i + 1)
    root.right = list_to_tree_dfs(arr, 2 * i + 2)
    return root


def level_order(root):
    # 广度优先遍历 (Breadth-first Traversal)
    queue = deque()
    queue.append(root)
    result = []
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)
    return result


root = list_to_tree_dfs(arr=[1, 2, 3, 4, 5, 6, 7], i=0)
result = level_order(root)
print(result)
