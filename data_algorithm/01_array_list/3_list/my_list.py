
class MyList:
    def __init__(self):
        # 受保护属性
        self._capacity = 10
        self._arr = [0] * self._capacity
        self._size = 0
        self._extend_ratio = 2
    
    def size(self):
        return self._size
    
    def capacity(self):
        return self._capacity
    
    def get(self, index):
        if index < 0 or index >= self._size:
            raise IndexError('Index error')
        return self._arr[index]
    
    def set(self, num: int, index: int):
        if index < 0 or index >= self._size:
            raise IndexError('Index error')
        self._arr[index] = num
    
    def add(self, num: int):
        # 元素数量超出容量时，触发扩容机制
        if self.size() == self.capacity():
            self.extend_capacity()
        self._arr[self._size] = num
        self._size += 1

    def insert(self, num: int, index: int):
        if index < 0 or index >= self._size:
            raise IndexError('Index error')
        if self._size == self.capacity():
            self.extend_capacity()
        # 将索引 index 以及以后的元素都向后移动一位
        for j in range(self._size - 1, index - 1, -1):
            self._arr[j+1] = self._arr[j]
        self._arr[index] = num
        self._size += 1

    def remove(self, index):
        if index < 0 or index >= self._size:
            raise IndexError('Index Error')
        num = self._arr[index]
        # 将索引 index 之后的元素都向前移动一位
        for j in range(index, self._size - 1):
            self._arr[j] = self._arr[j+1]
        # 更新元素数量
        self._size -= 1
        # 返回被删除的元素
        return num
    
    def extend_capacity(self):
        self._arr = self._arr + [0] * self.capacity() * (self._extend_ratio - 1)
    
    def to_array(self):
        return self._arr[: self._size]
