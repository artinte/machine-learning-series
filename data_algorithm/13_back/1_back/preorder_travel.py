class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left: TreeNode | None = None
        self.right: TreeNode | None = None


def list_to_tree_dfs(arr: list[int], i: int) -> TreeNode | None:
    # 使用递归将列表反序列化为二叉树
    # 如果索引超出数组长度，或者对应的元素为 None ，则返回 None
    if i < 0 or i >= len(arr) or arr[i] is None:
        return None
    root = TreeNode(arr[i])
    # 递归构建左右子树
    root.left = list_to_tree_dfs(arr, 2 * i + 1)
    root.right = list_to_tree_dfs(arr, 2 * i + 2)
    return root


result = list[TreeNode]()


def pre_order(root: TreeNode):
    if root is None:
        return
    if root.val == 7:
        result.append(root)
    pre_order(root.left)
    pre_order(root.right)


root = list_to_tree_dfs([1, 7, 3, 4, 5, 6, 7], 0)
pre_order(root)
print([node.val for node in result])
