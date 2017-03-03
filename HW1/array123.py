import unittest

"""
function implementing the http://codingbat.com/prob/p193604 exercise
"""
def array123(nums):
    return '1, 2, 3' in str(nums)

class TestFrontTimes(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(array123([1, 1, 2, 3, 4]))
        self.assertFalse(array123([1, 2, 4, 5]))

    def test_short(self):
        self.assertFalse(array123([1, 2]))
        self.assertFalse(array123([]))

if __name__ == '__main__':
    unittest.main()