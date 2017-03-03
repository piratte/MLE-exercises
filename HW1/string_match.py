import unittest

"""
function implementing the http://codingbat.com/prob/p182141 exercise
"""
def string_match(a, b):
    return sum([1 for i in range(0, min([len(a), len(b)])-1) if a[i:i+2] == b[i:i+2]])

class TestFrontTimes(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(string_match('xxcaazz', 'xxbaaz'), 3)
        self.assertEqual(string_match('abc', 'abc'), 2)
        self.assertEqual(string_match('he', 'hello'), 1)
        self.assertEqual(string_match('hello', 'he'), 1)

    def test_empty(self):
        self.assertEqual(string_match('', 'hello'), 0)


if __name__ == '__main__':
    unittest.main()