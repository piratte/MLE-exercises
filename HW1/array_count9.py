import unittest

"""
function implementing the http://codingbat.com/prob/p166170 exercise
"""
def array_count9(nums):
  return len(list(filter(lambda x: x == 9, nums)))

class TestFrontTimes(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(array_count9([1, 2, 9]), 1)
        self.assertEqual(array_count9([1, 9, 9]), 2)
        self.assertEqual(array_count9([1, 9, 9, 3, 9]), 3)

    def test_empty(self):
        self.assertEqual(array_count9([]), 0)


if __name__ == '__main__':
    unittest.main()