from collections import deque

class LRUCache:
    def __init__(self, capacity=30):
        self.dict1 = {}
        self.que = deque()
        self.capacity = capacity

    def push(self, key, val):
        if key in self.dict1:
            self.que.remove(key)
        elif len(self.dict1) >= self.capacity:
            # pop
            old_key = self.que.popleft()
            del self.dict1[old_key]
        
        self.dict1[key] = val
        self.que.append(key)

    def get(self, key):
        if key not in self.dict1:
            raise ValueError
        self.que.remove(key)
        self.que.append(key)
        return self.dict1[key]
