class TreeNode:
    def __init__(self, value=0, left=None, right=None) -> None:
        self.value = value
        self.left = left
        self.right = right

def build_tree():
    nodes = [TreeNode(i) for i in range(1, 16)]
    for i in range(7):
        left_index = 2 * i + 1
        right_index = 2 * i + 2
        if left_index < 15:
            nodes[i].left = nodes[left_index]
        if right_index < 15:
            nodes[i].right = nodes[right_index]
    return nodes[0]


root = build_tree()


def print_tree(node):
    if node is not None:
        print(str(node.value), end=' ')
        if node.left is not None or node.right is not None:
            if node.left:
                print_tree(node.left)
            if node.right:
                print_tree(node.right)


print_tree(root)
