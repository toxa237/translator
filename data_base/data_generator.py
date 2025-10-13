import tensorflow as tf
import json
from sqlalchemy import create_engine, select, or_, and_, func
from sqlalchemy.orm import Session
from keras.utils import PyDataset
from typing import Callable, Union, List, Tuple
from tensorflow.python.training.saving.saveable_object_util import set_cpu0
from data_base.data_base_conf import EnglishPhrase, PortuguesePhrase, FrenchPhrase,\
    BelarusianPhrase, SpanishPhrase, ItalianPhrase, RussianPhrase, TrainDataset
from tokenizers import Tokenizer


class TranslationDataGenerator(PyDataset):
    def __init__(self, laungage_couples: Union[str, List[Tuple[str, str]]] = 'all',
                 batch_size = 32, split='train', tokenisers: Callable|None=None, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self._engine = self._create_engene()
        self._language_couples = self._create_laungage_couples(laungage_couples, split)
        self.tokenisers = tokenisers
        self._language_classes = {
            'en': EnglishPhrase,
            'pt': PortuguesePhrase,
            'fr': FrenchPhrase,
            'be': BelarusianPhrase,
            'es': SpanishPhrase,
            'it': ItalianPhrase,
            'ru': RussianPhrase
        }
    
    def __len__(self):
        query = select(func.count(TrainDataset.id)).where(self._language_couples)
        connection = self._engine.connect()
        result = connection.execute(query).scalar()
        connection.close()
        return result // self.batch_size  # type: ignore
    
    def __getitem__(self, index):
        query = select(TrainDataset).where(self._language_couples).offset(
            index * self.batch_size).limit(self.batch_size)
        
        with Session(self._engine) as session:
            result: list[TrainDataset] = session.scalars(query).all()  # type: ignore

            inp_lang = [i.input_language for i in result]
            out_lang = [i.output_language for i in result]
            input_phrase_ids = [i.input_phrase_id for i in result]
            output_phrase_ids = [[int(j) for j in i.output_phrase_id.split('|')] for i in result]

            inp_lang_set = set(inp_lang)
            out_lang_set = set(out_lang)

            input_phrases_dict = {}
            for lang in inp_lang_set:
                ids = [pid for pid, l in zip(input_phrase_ids, inp_lang) if l == lang]  # type: ignore
                table = self._language_classes[lang]  # type: ignore
                phrases = session.execute(select(table).where(table.id.in_(ids))).scalars().all()
                input_phrases_dict[lang] = {p.id: p.phrase.lower() for p in phrases}
            
            output_phrases_dict = {}
            for lang in out_lang_set:
                all_ids = [pid for pids, l in zip(output_phrase_ids, out_lang) if l == lang for pid in pids]  # type: ignore
                table = self._language_classes[lang]  # type: ignore
                phrases = session.execute(select(table).where(table.id.in_(all_ids))).scalars().all()
                output_phrases_dict[lang] = {p.id: p.phrase.lower() for p in phrases}

        input_phrase = [input_phrases_dict[i.input_language][i.input_phrase_id] for i in result]

        output_phrase = []
        for out_ps, ot_l in zip(output_phrase_ids, out_lang):
            output_phrase.append(
                set([output_phrases_dict[ot_l][pid] for pid in out_ps])
            )
        output_phrase = ['[START]' + '[NEXT]'.join(phrases) + '[END]' for phrases in output_phrase]

        if self.tokenisers:
            input_phrase = self.tokenisers(inp_lang, input_phrase)
            output_phrase = self.tokenisers(out_lang, output_phrase)
            input_phrase = tf.constant(input_phrase, dtype=tf.string)
            output_phrase = tf.constant(output_phrase, dtype=tf.string)
        else:
            input_phrase = tf.constant(input_phrase, dtype=tf.int32)
            output_phrase = tf.constant(output_phrase, dtype=tf.int32)

        inp_lang = tf.constant(inp_lang, dtype=tf.string)
        out_lang = tf.constant(out_lang, dtype=tf.string)
        
        return {
                "inp_lang": inp_lang,
                "out_lang": out_lang,
                "input_phrase": input_phrase, 
                "output_phrase": output_phrase
               }
    
    def _create_engene(self):
        with open("configuration/config.json", "r") as f:
            db_patam = json.load(f)["db"]
            sql_engine = create_engine(
                f"{db_patam['server']}://{db_patam['user']}:{db_patam['password']}"
                f"@{db_patam['host']}:{db_patam['port']}/{db_patam['data_base']}"
            )
        return sql_engine
    
    def _create_laungage_couples(
            self, 
            language_couples: Union[str, List[Tuple[str, str]]],
            split
        ) -> or_:  # type: ignore
        self.unique_language_list = list(set([j for i in language_couples for j in i]))

        connection = self._engine.connect()
        posible_couples = select(TrainDataset.input_language,
                                 TrainDataset.output_language).distinct()
        posible_couples = connection.execute(posible_couples).fetchall()
        connection.close()

        if language_couples == 'all':
            language_couples = posible_couples  # type: ignore
        
        filters = []
        for lang1, lang2 in language_couples:
            if (lang1, lang2) not in posible_couples:
                raise ValueError(f"Unsupported language pair: {lang1}, {lang2}"
                                 f"possible pairs are: {posible_couples}")
            filters.append(and_(
                TrainDataset.input_language == lang1,
                TrainDataset.output_language == lang2
            ))
        split = True if split=='train' else False
        return and_(or_(*filters), TrainDataset.validation == split)
    

if __name__ == "__main__":
    q = TranslationDataGenerator(batch_size=2, laungage_couples=[('en', 'pt')])
    print(q[0])
    q = TranslationDataGenerator(batch_size=2, laungage_couples=[('en', 'pt')], split='test')
    print(q[0])

