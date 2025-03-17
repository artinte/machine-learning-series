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
    
    def left_rotate(self, node):
        child = node.right
        grand_child = child.left
        child.left = node
        node.right = grand_child
        self.update_height(node)
        self.update_height(child)
        return child

    def rotate(self, node):
        # 执行旋转操作，使该子树重新恢复平衡
        # 获取节点 node 的平衡因子
        balance_factor = self.balance_factor(node)
        # 左偏树
        if balance_factor > 1:
            if self.balance_factor(node.left) >= 0:
                # 右旋
                return self.right_rotate(node)
            else:
                # 先左旋后右旋
                node.left = self.left_rotate(node.left)
                return self.right_rotate(node)
        # 右偏树
        elif balance_factor < -1:
            if self.balance_factor(node.right) <= 0:
                # 左旋
                return self.left_rotate(node)
            else:
                # 先右旋后左旋
                node.right = self.right_rotate(node.right)
                return self.left_rotate(node)
        # 平衡树，无须旋转，直接返回
        return node
    
    def insert(self, val):
        self.root = self.recur_insert(self.root, val)
    
    def recur_insert(self, node, val):
        if node is None:
            return TreeNode(val)
        if val < node.val:
            node.left = self.recur_insert(node.left, val)
        elif val > node.val:
            node.right = self.recur_insert(node.right, val)
        else:
            return node
        self.update_height(node)
        return self.rotate(node)

    