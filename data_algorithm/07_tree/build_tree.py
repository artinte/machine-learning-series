class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def buildTree(self, preorder, inorder):
        if not preorder or not inorder:
            return None
        
        rootVal = preorder[0]
        root = TreeNode(rootVal)

        # 在中序遍历中找到根节点的位置
        rootIndex = inorder.index(rootVal)
        
        root.left = self.buildTree(preorder[1: 1 + rootIndex], inorder[:rootIndex])
        root.right = self.buildTree(preorder[rootIndex + 1:], inorder[rootIndex + 1:])

        return root
