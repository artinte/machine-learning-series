
lst = [1, 2, 3]
lst.append(4)
assert lst == [1, 2, 3, 4]

lst = [1, 2, 3]
lst.extend([4, 5])
assert lst == [1, 2, 3, 4, 5]

lst = [1, 2, 4]
lst.insert(2, 3)
assert lst == [1, 2, 3, 4]

lst = [1, 2, 3, 4]
last_item = lst.pop()   # 删除最后一个元素，和 append 对应
assert lst == [1, 2, 3]
assert last_item == 4

lst = [1, 2, 3, 2]
lst.remove(2)
assert lst == [1, 3, 2]

lst = [1, 2, 3]
lst.clear()
assert lst == []

lst = [10, 20, 30, 40, 50]
assert lst.index(20) == 1

lst = [1, 2, 2, 3, 2, 4]
assert lst.count(2) == 3

lst = [3, 1, 4, 1, 5, 9]
lst.sort()
assert lst == [1, 1, 3, 4, 5, 9]
lst.sort(reverse=True)
assert lst == [9, 5, 4, 3, 1, 1]

squares = [x**2 for x in range(5)]
assert squares == [0, 1, 4, 9, 16]

a = [1, 2, 3]
b = ['a', 'b', 'c']
zipped = list(zip(a, b))
assert zipped == [(1, 'a'), (2, 'b'), (3, 'c')]

numbers, letters = zip(*zipped)
assert numbers == (1, 2, 3)
assert letters == ('a', 'b', 'c')

lst = ['a', 'b', 'c']
for index, value in enumerate(lst):
    print(index, value)
