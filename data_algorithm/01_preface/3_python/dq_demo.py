from collections import deque

dq = deque(['a', 'b', 'c'])
print(dq)

dq.append('d')      # 从右侧添加元素
dq.appendleft('z')  # 从左侧添加元素
dq.pop()            # 从右侧弹出元素
dq.popleft()        # 从左侧弹出元素

# 设定队列容量
dq = deque(maxlen=3)
