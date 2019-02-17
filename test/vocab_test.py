import unittest
import torch

from vocab import Vocab


class VocabTest(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        return super().setUp()

    def test_to_input_sensor_char(self):
        sents = [['abc','12345'],['an'],['d', 'e', 'f', 'g']]
        vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json') 
        t = vocab.src.to_input_tensor_char(sents, self.device)
        # check that the (max_sentence_length, batch_size, max_word_length)
        max_word_length = 21
        batch_size = len(sents)
        max_sentence_length = 4
        self.assertEqual(list(t.shape), [max_sentence_length, batch_size,max_word_length])