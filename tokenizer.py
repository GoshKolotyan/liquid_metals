import json
import torch
from helper import parse_compostion


class LM_Tokenizer:
    def __init__(self):
        self.vocab = json.load(open("elements_vocab.json"))
        self.vocab_size = len(self.vocab)

    

    def encode(self, composition:str, target:float) -> torch.Tensor:
        """"
        encoding should be in fixed size input max size is 4 components
        output must be a tensot size of (4, n) where n is number of components
        """
        elements_fractions = parse_compostion(composition)
        output = []
        for element, fraction in elements_fractions:
            output.append([
                self.vocab[element]['atomic_number'] if element in self.vocab else 0,
                fraction,
                self.vocab[element]["atomic radius"] if element in self.vocab else 0,
                self.vocab[element]['electronegativity'] if element in self.vocab else 0,
                self.vocab[element]['melting_point'] if element in self.vocab else 0,
            ])
        return torch.tensor(output, dtype=torch.float64), \
               torch.tensor([target], dtype=torch.float64)

LM_Tokenizer = LM_Tokenizer()
composition , target = LM_Tokenizer.encode("Sn14.0Bi35.9In50.1", 0.1)
print(composition)
print(composition.shape)
print(target.item())






