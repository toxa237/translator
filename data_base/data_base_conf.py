from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, VARCHAR, TEXT, ForeignKey
import json


class Base(DeclarativeBase):
    pass


class Base_Phrase_Model(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    phrase = Column(TEXT, nullable=True)


class EnglishPhrase(Base_Phrase_Model):
    __tablename__ = 'english_phrases'

class PortuguesePhrase(Base_Phrase_Model):
    __tablename__ = 'portuguese_phrases'

class FrenchPhrase(Base_Phrase_Model):
    __tablename__ = 'french_phrases'

class BelarusianPhrase(Base_Phrase_Model):
    __tablename__ = 'belarusian_phrases'

class SpanishPhrase(Base_Phrase_Model):
    __tablename__ = 'spanish_phrases'

class ItalianPhrase(Base_Phrase_Model):
    __tablename__ = 'italian_phrases'

class RussianPhrase(Base_Phrase_Model):
    __tablename__ = 'russian_phrases'

class TrainDataset(Base):
    __tablename__ = 'train_dataset'

    id = Column(Integer, primary_key=True, autoincrement=True)
    input_language = Column(VARCHAR(2), nullable=False)
    output_language = Column(VARCHAR(2), nullable=False)
    input_phrase_id = Column(Integer, nullable=False)
    output_phrase_id = Column(VARCHAR(200), nullable=False)


if __name__ == "__main__":
    with open("configuration/config.json", "r") as f:
        db_param = json.load(f)["db"]
        sql_engine = create_engine(
            f"{db_param['server']}://{db_param['user']}:{db_param['password']}"
            f"@{db_param['host']}:{db_param['port']}/{db_param['data_base']}"
        )
    
    Base.metadata.drop_all(sql_engine)
    Base.metadata.create_all(sql_engine)

