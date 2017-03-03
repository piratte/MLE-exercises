import unittest

"""
function implementing the http://codingbat.com/prob/p119308 exercise
"""
def has22(nums):
    return '2, 2' in str(nums)

class TestFrontTimes(unittest.TestCase):
    def test_basic(self):
        self.assertFalse(has22([2, 1, 2]))
        self.assertTrue(has22([1, 2, 2]))
        self.assertTrue(has22([4, 2, 4, 2, 2, 5]))

    def test_empty(self):
        self.assertFalse(has22([]))

if __name__ == '__main__':
    unittest.main()
