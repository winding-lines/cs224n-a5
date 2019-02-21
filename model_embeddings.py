#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch.nn.functional as F

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

# Size of the char embeddings
E_CHAR=50

# Max input size
MAX_CHARS=21

# kernel
KERNEL = 5

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab['<pad>']
        vocab_len = len(vocab)
        self.embeddings = nn.Embedding(vocab_len, E_CHAR, padding_idx=pad_token_idx)
        self.cnn = CNN(in_channel=E_CHAR, out_channels=embed_size)
        self.maxpool = nn.MaxPool1d(MAX_CHARS-KERNEL+1)
        self.highway = Highway(in_features=embed_size, out_features=embed_size)
        self.dropout = nn.Dropout(0.3)


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        x1 = self.embeddings(input)
        x1_shaped = x1.view(x1.shape[0]*x1.shape[1], x1.shape[2], x1.shape[3])
        x1t =  x1_shaped.transpose(1,2)

        x2 = self.cnn(x1t)
        x_conv_out = self.maxpool(F.relu(x2))
        x3 = self.highway(x_conv_out.squeeze())
        x4 = self.dropout(x3)

        return x4.view(x1.shape[0], x1.shape[1], -1)
        ### END YOUR CODE

