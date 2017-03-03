import unittest

"""
function implementing the http://codingbat.com/prob/p193507 exercise
"""
def string_times(in_str, n):
  return "".join([in_str for i in range(0,n)])

class TestFrontTimes(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(string_times('Hi', 3), 'HiHiHi')
        self.assertEqual(string_times('Hi', 2), 'HiHi')
        self.assertEqual(string_times('x', 4), 'xxxx')

    def test_empty(self):
        self.assertEqual(string_times('', 3), '')

if __name__ == '__main__':
    unittest.main()