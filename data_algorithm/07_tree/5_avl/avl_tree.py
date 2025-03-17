class TreeNode:
    def __init__(self, val):
        self.val = val
        # 叶子节点的高度为 0，而空节点的高度为 -1
        self.height = 0
        self.left = None
        self.right = None

    def height(self, node):
        if node:
            return node.height
        return -1

    def update_height(self, node):
        node.height = max(self.height(node.left), self.height(node.right)) + 1

    def balance_factor(self, node):
        if node == None:
            return 0

        return self.height(node.left) - self.height(node.right)

    def right_rotate(self, node):
        child = node.left
        grand_child = child.right
        # 以 child 为原点，将 node 向右旋转
        child.right = node
        node.left = grand_child
        self.update_height(node)
        self.update_height(child)
        return child
    