def bubble_sort(nums: list[int]):
    n = len(nums)
    # 外循环：未排序未 [0, i]
    for i in range(n - 1, 0, -1):
        # 内循环：将未排序区间 [0, i] 中的最大元素交换至该区间的最右端
        for j in range(i):
            if nums[j] > nums[j + 1]:
                # 交换
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    return nums


print(bubble_sort([6, 4, 7, 5, 1, 8, 3, 9, 2]))
