from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 层次遍历获取右边节点
def right_side_view(root):
    if root == None:
        return []
    
    q = deque()
    q.append(root)
    result = []

    while q:
        length = len(q)
        
        for i in range(length):
            node = q.popleft()

            if length - 1 == i:
                result.append(node.val)

            if node.left:
                q.append(node.left)
            
            if node.right:
                q.append(node.right)

    return result

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.right = TreeNode(5)
root.right.right = TreeNode(4)

print(right_side_view(root))
