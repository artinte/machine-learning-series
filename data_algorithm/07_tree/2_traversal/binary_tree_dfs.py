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


result = []


def pre_order(root):
    # 前序遍历
    if root is None:
        return None
    result.append(root.val)
    # 访问优先级：根节点 - 左子树 - 右子树
    pre_order(root.left)
    pre_order(root.right)


def in_order(root):
    # 中序遍历
    if root is None:
        return None
    # 访问优先级：左子树 - 根节点 - 右子树
    in_order(root.left)
    result.append(root.val)
    in_order(root.right)


def post_order(root):
    # 后序遍历
    if root is None:
        return
    # 访问优先级：左子树 - 右子树 - 根节点
    post_order(root.left)
    post_order(root.right)
    result.append(root.val)


root = list_to_tree_dfs([1, 2, 3, 4, 5, 6, 7], 0)
pre_order(root)
print(result)
result.clear()

in_order(root)
print(result)
result.clear()

post_order(root)
print(result)
result.clear()
