from typing import List
from tokenizers import Tokenizer
import tensorflow as tf


class MultyLanguageTokenizer:
    def __init__(self, tokenizer_model, unique_language_list, out_size) -> None:
        self.unique_language_list = unique_language_list
        self.tokenizer_model = tokenizer_model
        self.out_size = out_size
        self.tokenizers = self._load_tokenizers()

    def encode(self, phrases: List[str], langs: List[str]):
        resault = []
        for phr, lang in zip(phrases, langs):
            res = self.tokenizers[lang].encode(phr).ids
            res = res + [0] * (self.out_size - len(res)) if len(res) < self.out_size\
                else res[:self.out_size]
            resault.append(res)
        return resault

    def train(self, inputs):
        inputs['input_phrase'] = self.encode(inputs['input_phrase'], inputs['inp_lang'])
        inputs['output_phrase'] = self.encode(inputs['output_phrase'], inputs['out_lang'])
        return inputs, tf.int32

    def decode(self, phrases: List[str], langs: List[str]):
        resault = []
        for phr, lang in zip(phrases, langs):
            res = self.tokenizers[lang].decode(phr)
            resault.append(res)
        return resault
        
    def _load_tokenizers(self):
        out = {}
        for lang in self.unique_language_list:
            path = f"tokenizers_modeles/{self.tokenizer_model}/{lang}.json"
            out[lang] = Tokenizer.from_file(path)
        return out

