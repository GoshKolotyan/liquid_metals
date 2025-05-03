import json
import torch
from helper import parse_compostion, LM_feature_extractor


class LM_Tokenizer:
    def __init__(self):
        self.vocab = json.load(open("./Data/elements_vocab.json"))
        self.vocab_size = len(self.vocab)
        self.feature_extractor = LM_feature_extractor()

    def encode(self, composition:str, target:float) -> torch.Tensor:
        """"
        encoding should be in fixed size input max size is 4 components
        output must be a tensot size of (5, n) where n is number of components
        """
        elements_fractions = parse_compostion(composition)
        output = []
        calculate_ = self.feature_extractor.__call__(elements_fractions)
        for element, fraction in elements_fractions:
            output.append([
                self.vocab[element]['atomic_number'] if element in self.vocab else 0,
                fraction,
                self.vocab[element]["atomic radius"] if element in self.vocab else 0,
                self.vocab[element]['electronegativity'] if element in self.vocab else 0,
                self.vocab[element]['melting_point'] if element in self.vocab else 0,
            ])
        output.append(calculate_)

        return torch.tensor(output, dtype=torch.float32), torch.tensor([target], dtype=torch.float32)


# # Encode composition
# LM_Tokenizer = LM_Tokenizer()
# composition = "Sn14.0Bi36In25.0Ag25.0"
# print("Composition:", composition, "Target:", 0.1)
# composition, target = LM_Tokenizer.encode(composition=composition, target=0.1)
# print("Original composition shape after tokenzier:", composition.shape)
# print("Composition tokenized:\n", composition)
# print("Target:", target.item())
