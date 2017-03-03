import unittest

"""
function implementing the http://codingbat.com/prob/p184853 exercise
"""
def big_diff(nums):
    try:
        return reduce(max, nums) - reduce(min, nums)
    except NameError: # reduce not available in python3
        maxim = nums[0]
        minim = nums[0]
        for num in nums:
            minim = num if num < minim else minim
            maxim = num if num > maxim else maxim
        return maxim - minim

class TestFrontTimes(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(big_diff([10, 3, 5, 6]), 7)
        self.assertEqual(big_diff([7, 7, 6, 8, 5, 5, 6]), 3)
        self.assertEqual(big_diff([5, 1, 6, 1, 9, 9]), 8)

    def test_small(self):
        self.assertEqual(big_diff([6]), 0)

if __name__ == '__main__':
    unittest.main()