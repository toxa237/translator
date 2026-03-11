import os
from tokenizers import Tokenizer 
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, Strip, Sequence
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import json

from data_base.data_base_conf import EnglishPhrase, PortuguesePhrase, \
    FrenchPhrase, BelarusianPhrase, SpanishPhrase, ItalianPhrase, RussianPhrase


tokenizer_model, tokenizer_trainer = BPE, BpeTrainer 
vocab_size = 10000

model_name = f"tokenizers_modeles/tokenizer_model_{tokenizer_model.__name__}_{vocab_size}"
if not os.path.isdir(model_name):
    os.mkdir(model_name)

spesial_tokens = ["[UNK]", "[START]", "[NEXT]", "[STOP]"]
language_codes = {
    'pt': PortuguesePhrase,
    'en': EnglishPhrase,
    'be': BelarusianPhrase,
    'es': SpanishPhrase,
    'fr': FrenchPhrase,
    'it': ItalianPhrase,
    'ru': RussianPhrase
}

with open("configuration/symbols.json") as f:
    alfabets: dict = json.load(f)

with open("configuration/config.json", "r") as f:
    db_patam = json.load(f)["db"]
    sql_engine = create_engine(
        f"{db_patam['server']}://{db_patam['user']}:{db_patam['password']}"
        f"@{db_patam['host']}:{db_patam['port']}/{db_patam['data_base']}"
    )


for lang in language_codes:
    with Session(sql_engine) as session:  # type: ignore
        df = session.execute(select(language_codes[lang].phrase)).all()
    df = [i[0] for i in df]
    tokenizer = Tokenizer(tokenizer_model())

    tokenizer.normalizer = Sequence([Lowercase(), Strip()])  # type: ignore
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore

    trainer = tokenizer_trainer(vocab_size=vocab_size,
                                special_tokens=spesial_tokens,
                                initial_alphabet=list(alfabets[lang]) + [" "]
                                )
    tokenizer.train_from_iterator(iterator=df, trainer=trainer)
    print([{i: tokenizer.encode(i).ids} for i in alfabets[lang]])
    print(tokenizer.get_vocab_size())
    tokenizer.save(f"{model_name}/{lang}.json")


