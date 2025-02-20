
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next: ListNode | None = None

class LinkedListStack:
    def __init__(self):
        self._peek: ListNode | None = None
        self._size = 0
    
    def size(self):
        return self._size

    def is_empty(self):
        return self._size == 0
    
    def push(self, val):
        node = ListNode(val)
        node.next = self._peek
        self._peek = node
        self._size += 1
    
    def peek(self):
        if self.is_empty():
            raise IndexError('Index Error')
        if self._peek:
            return self._peek.val
        else:
            raise IndexError('Index Error')
    
    def pop(self):
        num = self.peek()
        if self._peek:
            self._peek = self._peek.next
        else:
            raise IndexError('Index Error')
        self._size -= 1
        return num
    
    def to_list(self):
        arr = []
        node = self._peek
        while node:
            arr.append(node.val)
        arr.reverse()
        return arr
    