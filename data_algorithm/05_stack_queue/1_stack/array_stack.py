class ArrayStack:
    '''
    基于数组实现的栈
    '''
    def __init__(self):
        self._stack: list[int] = []

    def size(self):
        return len(self._stack)

    def empty(self):
        return self.size() == 0

    def push(self, item):
        self._stack.append(item)

    def pop(self):
        if self.empty():
            raise IndexError('Index error')
        return self._stack.pop()

    def peek(self):
        if self.empty():
            raise IndexError('Stack is empty')
        return self._stack[-1]

    def to_list(self):
        return self._stack