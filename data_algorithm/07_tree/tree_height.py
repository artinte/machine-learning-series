class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
def treeHeight(root):
    if root is None:
        # 叶子节点高度为 0，空树高度设为 -1
        return -1

    left_height = treeHeight(root.left)
    right_height = treeHeight(root.right)
    
    return max(left_height, right_height) + 1
