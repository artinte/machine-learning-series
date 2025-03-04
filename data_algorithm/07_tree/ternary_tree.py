class TreeNode:
    def __init__(self, val=0, left=None, center=None, right=None):
        self.val = val
        self.left = left
        self.center = center
        self.right = right


def insert(root, val):
    if root is None:
        return TreeNode(val)
    if val < root.val - 500:
        root.left = insert(root.left, val)
    elif val > root.val + 500:
        root.right = insert(root.right, val)
    else:
        root.center = insert(root.center, val)
    return root

def get_height(root):
    if root is None:
        return 0
    left_height = get_height(root.left)
    center_height = get_height(root.center)
    right_height = get_height(root.right)
    return max(left_height, center_height, right_height) + 1


nums = [5000, 2000, 5000, 8000, 1800]

root = TreeNode(5000)
for i in range(1, len(nums)):
    insert(root, nums[i])

print(get_height(root))
