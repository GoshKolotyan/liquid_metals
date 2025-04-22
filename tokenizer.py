import re
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from helper import parse_compostion



class LM_Tokenizer:
    def __init__(self):
        self.vocab = None
        self.vocab_size = len(self.vocab)
    

    def encode(self, composition:str, target:float) -> torch.Tensor:
        """"
        encoding should be in fixed size input max size is 4 components
        output must be a tensot size of (4, n) where n is number of components
        """
        elements_fractions = parse_compostion(composition)




