 
arr = [0] * 5
assert arr == [0, 0, 0, 0, 0]

nums: list[int] = [1, 3, 2, 5, 4]
assert nums == [1, 3, 2, 5, 4]


def insert(nums, num, index):
    # 插入
    for i in range(len(nums) - 1, index, -1):
        nums[i] = nums[i - 1]
    nums[index] = num
    
insert(nums, 6, 2)
assert nums == [1, 3, 6, 2, 5]

