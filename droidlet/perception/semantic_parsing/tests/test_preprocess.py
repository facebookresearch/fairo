import unittest
from ..utils.preprocess import preprocess_chat, insert_spaces


class MyTestCase(unittest.TestCase):
    def test_insert_spaces(self):
        updated_chat = insert_spaces("make a 2x2 cube")
        self.assertEqual(updated_chat, "make a 2 x 2 cube")
        updated_chat = insert_spaces("move the chair to 1,2,3")
        self.assertEqual(updated_chat, "move the chair to 1 , 2 , 3")

    def test_preprocess_chat(self):
        self.assertEqual(preprocess_chat("put cup at 0,0,0"), "put cup at 0 , 0 , 0")
        self.assertEqual(preprocess_chat("hi there! how are you"), "hi there ! how are you")


if __name__ == '__main__':
    unittest.main()
