import unittest

"""
function implementing the http://codingbat.com/prob/p165097 exercise
"""
def front_times(str, n):
  res = ''
  prefix = str[:3]
  for i in range(0,n): res += prefix
  return res

class TestFrontTimes(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(front_times('Chocolate', 2), 'ChoCho')
        self.assertEqual(front_times('Abc', 3), 'AbcAbcAbc')

    def test_short(self):
        self.assertEqual(front_times('A', 4), 'AAAA')

    def test_empty(self):
        self.assertEqual(front_times('', 4), '')

    def test_norep(self):
        self.assertEqual(front_times('Abc', 0), '')

if __name__ == '__main__':
    unittest.main()