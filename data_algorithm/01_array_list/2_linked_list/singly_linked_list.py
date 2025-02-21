class SinglyListNode:
    def __init__(self, val: int):
        self.val: int = val
        self.next: SinglyListNode | None = None


n0 = SinglyListNode(1)
n1 = SinglyListNode(2)
n2 = SinglyListNode(3)
n3 = SinglyListNode(4)
n4 = SinglyListNode(5)

n0.next = n1
n1.next = n2
n2.next = n3
n3.next = n4


def print_list(iter_node):
    while iter_node != None:
        print(iter_node.val, end=" ")
        iter_node = iter_node.next
    print()


print_list(n0)


def insert(node: SinglyListNode, inserted: SinglyListNode):
    # 在链表的节点 node 之后插入节点 inserted
    temp = node.next
    node.next = inserted
    inserted.next = temp


inserted = SinglyListNode(6)
insert(n2, inserted)
print_list(n0)


def remove(node: SinglyListNode):
    if not node.next:
        return
    node.next = node.next.next


remove(n2)
print_list(n0)


def access(head: SinglyListNode | None, index: int) -> SinglyListNode | None:
    for _ in range(index):
        if not head:
            return None
        head = head.next
    return head


n3 = access(n0, 3)
if n3:
    print(n3.val)


def find(head: SinglyListNode | None, target: int) -> int:
    # 在链表中查找值为 target 的首个节点
    index = 0
    while head:
        if head.val == target:
            return index
        head = head.next
        index += 1
    return -1


print(find(n0, 2))
