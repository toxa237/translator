import string
import json
import numpy as np
from typing import List
import tensorflow as tf


def vocab_creator(list_of_lang: List):
    punctuation = list(string.punctuation + string.digits + " ") + ['§', '¤', '¶']# ['<START>', '<NEXT>', '<END>']
    vocab = {v: k for k, v in enumerate(punctuation)}
    
    with open('configuration/symbols.json', 'r') as f:
        vocab_lang = [v for l, v in json.load(f).items() if l in list_of_lang]
    
    symbols_count = len(vocab)
    for symbols in vocab_lang:
        vocab.update({v: k + symbols_count for k, v in enumerate(symbols)})

    return np.array([i for i in vocab.keys()])

def custom_standardize(input_str):
    input_str = tf.strings.regex_replace(input_str, "§", '')
    input_str = tf.strings.regex_replace(input_str, "¤", '')
    input_str = tf.strings.regex_replace(input_str, "¶", '')
    input_str = tf.strings.regex_replace(input_str, r"<START>", "§")
    input_str = tf.strings.regex_replace(input_str, r"<NEXT>", "¤")
    input_str = tf.strings.regex_replace(input_str, r"<END>", "¶")
    return input_str
