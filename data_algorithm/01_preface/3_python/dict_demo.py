from collections import defaultdict

d = {
    'name': 'Alice',
    'age': 25,
    'city': 'Beijing'
}

assert d['name'] == 'Alice'
d['age'] = 26

del d['city']   # 删除键值对
age = d.pop('age')
assert age == 26

for key, value in d.items():
    print(key, value)

d = defaultdict(int)
d['a'] = 1
print(d)

