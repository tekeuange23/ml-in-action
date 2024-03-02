from module import capitalize
import unittest

class TestCap(unittest.TestCase):
    
    def empty(self) -> None:
        word = ''
        obtained = capitalize(word)
        expected = ''
        self.assertEqual(obtained, expected)
    
    def word_test(self) -> None:
        word = 'python'
        obtained = capitalize(word)
        expected = 'Python'
        self.assertEqual(obtained, expected)
        
    def words_test(self) -> None:
        words = 'python programming'
        obtained = capitalize(words)
        expected = 'Python Programmidng'
        self.assertEqual(obtained, expected)


if __name__ == '__main__':
    unittest.main()