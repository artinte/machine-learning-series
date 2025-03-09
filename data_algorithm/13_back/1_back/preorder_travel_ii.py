
result = []
path = []
def pre_order(root):
    if root is None:
        return
    
    # 尝试
    path.append(root)
    if root.val == 7:
        # 记录解
        result.append(path[:])
    pre_order(root.left)
    pre_order(root.right)
    # 回退
    path.pop()

    