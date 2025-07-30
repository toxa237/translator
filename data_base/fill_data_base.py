import pandas as pd
import tensorflow_datasets as tfds
from sqlalchemy import create_engine, select, insert
import json
from data_base.data_base_conf import EnglishPhrase, PortuguesePhrase, \
    FrenchPhrase, BelarusianPhrase, SpanishPhrase, ItalianPhrase, RussianPhrase, TrainDataset


language_codes = {
    'pt': PortuguesePhrase,
    'en': EnglishPhrase,
    'be': BelarusianPhrase,
    'es': SpanishPhrase,
    'fr': FrenchPhrase,
    'it': ItalianPhrase,
    'ru': RussianPhrase
}
list_of_datasets = ['pt_to_en', 'be_to_en', 'es_to_pt', 'fr_to_pt', 'it_to_pt',
                    'ru_to_en']
with open("configuration/config.json", "r") as f:
        db_patam = json.load(f)["db"]
        sql_engine = create_engine(
            f"{db_patam['server']}://{db_patam['user']}:{db_patam['password']}"
            f"@{db_patam['host']}:{db_patam['port']}/{db_patam['data_base']}"
        )


def bulk_get_or_create(phrases_in: pd.Series, language_class, connection):
    phrases = phrases_in.dropna().astype(str).unique()
    phrases_in.name = 'phrase'
    
    result = connection.execute(
        language_class.__table__.select().where(language_class.phrase.in_(phrases))
    ).fetchall()
    existing = {row.phrase: row.id for row in result}

    to_insert = [p for p in phrases if p not in existing]

    if to_insert:
        connection.execute(
            language_class.__table__.insert(),
            [{"phrase": p} for p in to_insert]
        )
        connection.commit()
    
    existing = pd.read_sql(
        select(language_class.phrase, language_class.id).where(
            language_class.phrase.in_(phrases)
        ),
        connection
    )
    phrases_in = pd.merge(phrases_in, existing, on='phrase', how='left')

    return phrases_in['id']


for dataset_name in list_of_datasets:
    print(f'Processing dataset: {dataset_name}')
    dataset = tfds.load(f'ted_hrlr_translate/{dataset_name}', split='train')
    df = tfds.as_dataframe(dataset)
    df = df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    df.to_sql(dataset_name, sql_engine, if_exists='replace', index=False)

    df = pd.read_sql(dataset_name, sql_engine)
    columns = df.columns
    connection = sql_engine.connect()

    for col in columns:
        df[f'{col}_id'] = bulk_get_or_create(df[col].copy(), language_codes[col], connection)
    connection.close()

    for src, tgt in [(0, 1), (1, 0)]:
        df_grouped = df.groupby(f'{df.columns[src]}_id')[f'{df.columns[tgt]}_id'].agg(
            lambda x: '|'.join(map(str, set(x)))
            )
        df_grouped = df_grouped.reset_index()
        df_grouped.columns = ['input_phrase_id', 'output_phrase_id']
        df_grouped['input_language'] = df.columns[src]
        df_grouped['output_language'] = df.columns[tgt]
        df_grouped.to_sql(TrainDataset.__tablename__, con=sql_engine,
                          if_exists='append', index=False)
        print(f'Inserted {len(df_grouped)} records for {df.columns[src]} to {df.columns[tgt]}')

