
s = {1, 2, 3, 4}
s = set([1, 2, 3, 4])
s = set()

s = {1, 2, 3}
s.add(4)
s.update([5, 6, 7]) # 添加多个元素
assert s == {1, 2, 3, 4, 5, 6, 7}

s.remove(2)     # 不存在会报错
s.discard(3)    # 不存在不会报错
assert s == {1, 4, 5, 6, 7}
s.pop()         # 随机移除
print(s)
