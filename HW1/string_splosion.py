import unittest

"""
function implementing the http://codingbat.com/prob/p118366 exercise
"""
def string_splosion(str):
  res = ''
  for i in range(0, len(str)):
    res += str[:i]
  return res + str


class TestStringSlosion(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(string_splosion('Code'), 'CCoCodCode')
        self.assertEqual(string_splosion('abc'), 'aababc')

if __name__ == '__main__':
    unittest.main()
