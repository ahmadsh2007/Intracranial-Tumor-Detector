class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        nums[:] = list(set(nums))
       return len(list(set(nums)))
    
solution = Solution()
nums = [0,0,1,1,1,2,2,3,3,4]
print(solution.removeDuplicates(nums))