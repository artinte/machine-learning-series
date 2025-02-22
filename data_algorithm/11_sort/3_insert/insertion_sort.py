def insertion_sort(nums: list[int]):
    # 外循环：已排序区间为 [0, i-1]
    for i in range(1, len(nums)):
        base = nums[i]
        j = i - 1
        # 内循环：将 base 插入到已排序区间 [0, i-1] 中的正确位置
        while j >= 0 and nums[j] > base:
            # 将 nums[j] 向右移动一位
            nums[j + 1] = nums[j]
            j -= 1
        # 将 base 赋值到正确位置
        nums[j + 1] = base
    return nums


print(insertion_sort([6, 4, 7, 5, 1, 8, 3, 9, 2]))
