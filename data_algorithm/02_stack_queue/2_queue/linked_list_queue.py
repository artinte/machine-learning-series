class ListNode:
    def __init__(self, val):
        self.val = val
        self.next: ListNode | None = None

class LinkedListQueue:
    def __init__(self):
        # 头节点
        self._front: ListNode | None = None
        # 尾节点
        self._rear: ListNode | None = None
        self._size = 0
    
    def size(self):
        return self.size
    
    def is_empty(self):
        return self._size == 0
    
    def push(self, num):
        node = ListNode(num)
        # 如果队列为空，令头尾节点都指向该节点
        if self._front is None:
            self._front = node
            self._rear = node
        else:
            # 如果队列不为空，则将该节点添加到尾节点
            if self._rear:
                self._rear.next = node
                self._rear = node
        self._size += 1

    def peek(self):
        if self.is_empty():
            raise IndexError('Index error')
        if self._front:
            return self._front.val

    def pop(self):
        num = self.peek()
        if self._front:
            self._front = self._front.next
        else:
            raise IndexError('Index Error')
        self._size -= 1
        return num
    
    def to_list(self):
        queue = []
        temp = self._front
        while temp:
            queue.append(temp.val)
            temp = temp.next
        return queue
