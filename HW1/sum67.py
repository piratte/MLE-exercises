import unittest

"""
function implementing the http://codingbat.com/prob/p108886 exercise
"""
def sum67(nums):
    cur_sum = 0
    off = False
    for num in nums:
        if num == 6: off = True
        elif num == 7:
            if not off: cur_sum += num
            off = False
        else: cur_sum = cur_sum + num if not off else cur_sum
    return cur_sum

class TestFrontTimes(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(sum67([1, 1, 6, 7, 2]), 4)
        self.assertEqual(sum67([2, 7, 6, 2, 6, 2, 7]), 9)
        self.assertEqual(sum67([6, 7, 1, 6, 7, 7]), 8)

    def test_empty(self):
        self.assertEqual(sum67([]), 0)

if __name__ == '__main__':
    unittest.main()