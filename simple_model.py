from typing import List
import tensorflow as tf
from keras import models, layers, optimizers, losses, callbacks
from keras.saving import serialize_keras_object, deserialize_keras_object
from tokenizers import Tokenizer
from data_base.data_generator import TranslationDataGenerator
from utils import castom_layers


class TextTokeniser:
    def __init__(self) -> None:
        pass

    def __call__(self):
        pass


def create_encoder_model(vocab_size: int, text_input_leight: int,
                         num_heads=8, key_dim=256) -> models.Model:
    input_layer = layers.Input(shape=(text_input_leight,), name='input_phrase')

    X = layers.Embedding(input_dim=vocab_size+2, output_dim=key_dim, name='embeding')(input_layer)

    masking_layer = castom_layers.PadMask()(input_layer)

    X = castom_layers.PositionalEncoding(max_len=text_input_leight, d_model=key_dim)(X)

    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
        )(
            X, X, X,
            attention_mask=masking_layer
        )
    X = layers.LayerNormalization()(attention_output + X)

    X_ffn = layers.Dense(key_dim, activation='relu')(X)
    X = layers.LayerNormalization()(X + X_ffn)
    X = layers.Dense(key_dim, activation='relu')(X)
    X = layers.Dense(key_dim, activation='relu')(X)
    model = models.Model(inputs=input_layer, outputs=X, name='encoder_model')
    return model


def crate_decoder_model(vocab_size: int, text_input_leight: int, text_output_leight: int|None=None,
                        num_heads=8, key_dim=256) -> models.Model:
    if text_output_leight is None:
        text_output_leight = text_input_leight
    input_embeding = layers.Input(shape=(800, key_dim), name='embeding')
    input_decoder = layers.Input(shape=(799,), name='decoder_input')

    emb_decoder = layers.Embedding(input_dim=vocab_size+2, output_dim=key_dim, name='emb_decoder')(input_decoder)

    emb_decoder = castom_layers.PositionalEncoding(max_len=800, d_model=key_dim)(emb_decoder)

    attention_mask = castom_layers.DecoderMask()(input_decoder)

    X = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
        )(emb_decoder, emb_decoder, emb_decoder, attention_mask=attention_mask)

    out_1 = layers.LayerNormalization()(X + emb_decoder)

    X = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
        )(out_1, input_embeding, input_embeding)
    out_2 = layers.LayerNormalization()(out_1 + X)

    X = layers.Dense(key_dim, activation='relu')(out_2)
    output_layer = layers.LayerNormalization()(out_2 + X)
    output_layer = layers.Dense(256, activation='relu')(output_layer)
    output_layer = layers.Dense(vocab_size + 2)(output_layer)
    
    model = models.Model(inputs=[input_embeding, input_decoder], outputs=output_layer)
    return model


class TranslationModel(models.Model):
    def __init__(self, encoder: models.Model, decoder: models.Model,
                 unique_language_list: List, tokens_model: str, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.tokenisers = self.get_dict_of_tokenasiers(unique_language_list, tokens_model)
    
    def compile(self, optimizer, loss, metrics=None):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def call(self, input_phrase, decoder_input, training=False):
        embedings = self.encoder(input_phrase, training=training)
        decoder_output = self.decoder([embedings, decoder_input], training=training)
        return decoder_output

    def train_step(self, inputs):
        input_phrase, output_phrases = inputs['input_phrase'], inputs['decoder_input']
        input_phrase = self.text_vector_inp(input_phrase)
        output_phrases = self.text_vector_out(output_phrases)
        decoder_input = output_phrases[:, :-1]
        target_output = output_phrases[:, 1:]
        with tf.GradientTape() as tape:
            predictions = self(input_phrase, decoder_input, training=True)
            loss = self.compute_loss(y=target_output, y_pred=predictions)

        gradients = tape.gradient(loss, self.trainable_variables)  # type:ignore
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))  # type:ignore

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(target_output, predictions)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        input_phrase, output_phrases = inputs['input_phrase'], inputs['decoder_input']
        input_phrase = self.text_vector_inp(input_phrase)
        output_phrases = self.text_vector_out(output_phrases)

        decoder_input = output_phrases[:, :-1]
        target_output = output_phrases[:, 1:]

        predictions = self(input_phrase, decoder_input, training=False)
        loss = self.compute_loss(y=target_output, y_pred=predictions)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(target_output, predictions)
 
        return {m.name: m.result() for m in self.metrics}

    def get_dict_of_tokenasiers(self, unique_language_list: List, tokens_model: str) -> tf.lookup.StaticHashTable:
        tokenisers = []
        for lang in unique_language_list:
            tokenisers.append(Tokenizer.from_file(f'tokenizers/{tokens_model}_tokens_model/{lang}.json'))
        tokenisers = tf.lookup.KeyValueTensorInitializer(unique_language_list, tokenisers)
        return tf.lookup.StaticHashTable(tokenisers, -1, 'tokenisers')
    
    def summary(self, *args, **kwargs):
        self.encoder.summary(*args, **kwargs)
        self.decoder.summary(*args, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder': serialize_keras_object(self.encoder),
            'decoder': serialize_keras_object(self.decoder),
            'unique_language_list': serialize_keras_object(self.tokenisers)
        })
        return config
    
    @classmethod
    def from_config(cls, config, *args, **kwargs):
        encoder = deserialize_keras_object(config['encoder'])
        decoder = deserialize_keras_object(config['decoder'])
        unique_language_list = deserialize_keras_object(config['unique_language_list'])
        instance = cls(encoder=encoder, decoder=decoder,
                       unique_language_list=unique_language_list)
        return instance
      

if __name__ == "__main__":
    batch_size = 8
    laungage_couples=[('en', 'pt')]
    num_heads = 8
    key_dim = 256
    vocab_size = 20_000
    model_name = 'model_t'
    data_generator = TranslationDataGenerator(laungage_couples, batch_size, 'train')
    data_generator_val = TranslationDataGenerator(laungage_couples, batch_size, 'validation')

    encoder_model = create_encoder_model(vocab_size, num_heads, key_dim)
    decoder_model = crate_decoder_model(vocab_size, num_heads, key_dim)
    
    translation_model = TranslationModel(encoder=encoder_model, decoder=decoder_model,
                                         unique_language_list=data_generator.unique_language_list)
    loss = losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)
    optimizer = optimizers.Adam()
    translation_model.compile(optimizer=optimizer, loss=loss)

    translation_model.summary()
    
    callback = [
            callbacks.ModelCheckpoint(
                filepath=f'models/{model_name}/' + 'translation_model_epoch_{epoch:02d}_loss_{loss:.4f}_val_loss'\
                    '_{val_loss:.4f}.keras',
                save_freq='epoch',
                save_weights_only=False,
                save_best_only=False,
                verbose=1
                )
            ]
    try:
        translation_model.fit(data_generator, epochs=5, callbacks=callback, validation_data=data_generator_val,
                              validation_steps=100)
    except KeyboardInterrupt:
        pass
    translation_model.save(f'models/{model_name}/translation_model_final.keras')

