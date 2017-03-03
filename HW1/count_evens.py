import unittest

"""
function implementing the http://codingbat.com/prob/p189616 exercise
"""
def count_evens(nums):
    return len([i for i in nums if i % 2 == 0])

class TestFrontTimes(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(count_evens([2, 1, 2, 3, 4]), 3)
        self.assertEqual(count_evens([11, 9, 0, 1]), 1)
        self.assertEqual(count_evens([2, 11, 9, 0]), 2)

    def test_empty(self):
        self.assertEqual(count_evens([]), 0)

if __name__ == '__main__':
    unittest.main()
