# 初始化哈希表

hmap = {}
# 在哈希表中添加键值对 (key, value)
hmap[123] = '鸭'
hmap[234] = '鸡'
hmap[345] = '猪'
hmap[456] = '牛'

print(hmap[345])
hmap.pop(345)
print(hmap)
