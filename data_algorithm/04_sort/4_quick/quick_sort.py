def partition(nums: list[int], left, right) -> int:
    # 以 nums[left] 为基准数
    i, j = left, right
    
    while i < j:
        while i < j and nums[j] >= nums[left]:
            # 从右向左找首个小于基准数的元素
            j -= 1
        while i < j and nums[i] <= nums[left]:
            # 从左向右找首个大于基准数的元素
            i += 1
        # 元素交换
        nums[i], nums[j] = nums[j], nums[i]
    # 将基准数交换到两子数组的分界线
    nums[i], nums[left] = nums[left], nums[i]
    # 返回基准数的索引
    return i

def quick_sort(nums: list[int], left, right):
    if left >= right:
        return
    # 哨兵划分
    pivot = partition(nums, left, right)
    quick_sort(nums, left, pivot - 1)
    quick_sort(nums, pivot + 1, right)

nums = [6, 4, 7, 5, 1, 8, 3, 9, 2]
quick_sort(nums, 0, len(nums) - 1)
print(nums)
