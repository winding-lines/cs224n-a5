import unittest
import utils

class UtilsTest(unittest.TestCase):
    def testPadSentsChar(self):
        sents = [['abc','12345'],['an']]
        padded = utils.pad_sents_char(sents,' ', 3)
        self.assertEqual(padded, [[['a','b','c'], ['1','2','3']], [['a','n',' '], [' ', ' ', ' ']]])