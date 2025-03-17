class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, num):
        if self.root == None:
            self.root = TreeNode(num, None, None)
            return
    
        # 查找节点，叶节点之后跳出
        pre, cur = None, self.root
        while cur:
            if cur.val == num:
                return
            pre = cur
            
            if cur.val < num:
                cur = cur.right
            else:
                cur = cur.left
        
        # 插入节点
        node = TreeNode(num)
        if pre.val < num:
            pre.right = node
        else:
            pre.left = node
    
bst = BinarySearchTree()
nums = [4, 2, 6, 1, 3, 5, 7]
for num in nums:
    bst.insert(num)

bst.insert(16)
