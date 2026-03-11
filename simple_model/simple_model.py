from pathlib import Path
import tensorflow as tf
from keras import models, layers, optimizers, losses, callbacks
from keras.saving import serialize_keras_object, deserialize_keras_object
from data_base.data_generator import TranslationDataGenerator
from tokenizers_modeles.tokenizer_prepr import MultyLanguageTokenizer
from utils import castom_layers


def create_encoder_model(vocab_size: int, text_leight: int,
                         num_heads=8, key_dim=256) -> models.Model:
    input_layer = layers.Input(shape=(text_leight,), name='input_phrase')
    
    X = layers.Embedding(input_dim=vocab_size + 2, output_dim=key_dim, name='embeding')(input_layer)
    X = castom_layers.PositionalEncoding(max_len=text_leight, d_model=key_dim)(X)

    masking_layer = castom_layers.PadMask()(input_layer)

    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
    )(X, X, attention_mask=masking_layer)
    attention_output = layers.Dropout(0.3)(attention_output)
    X = layers.LayerNormalization()(attention_output + X)

    X_ffn = layers.Dense(key_dim * 4, activation='gelu')(X)
    X_ffn = layers.Dense(key_dim)(X_ffn)
    X_ffn = layers.Dropout(0.3)(X_ffn)
    X = layers.LayerNormalization()(X + X_ffn)
    
    return models.Model(inputs=input_layer, outputs=[X, masking_layer], name='encoder_model')


def crate_decoder_model(vocab_size: int, text_leight: int,
                        num_heads=8, key_dim=256) -> models.Model:
    input_embeding = layers.Input(shape=(text_leight, key_dim), name='encoder_outputs')
    input_decoder = layers.Input(shape=(text_leight,), name='decoder_input')
    encoder_mask = layers.Input(shape=(1, 1, text_leight), name='encoder_mask')

    causal_mask = castom_layers.DecoderMask()(input_decoder)

    emb_decoder = layers.Embedding(input_dim=vocab_size + 2, output_dim=key_dim)(input_decoder)
    emb_decoder = castom_layers.PositionalEncoding(max_len=text_leight, d_model=key_dim)(emb_decoder)
    emb_decoder = layers.Dropout(0.3)(emb_decoder)

    X = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
        emb_decoder, emb_decoder, attention_mask=causal_mask
    )
    out_1 = layers.LayerNormalization()(X + emb_decoder)

    X = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
        out_1, input_embeding, input_embeding, attention_mask=encoder_mask
    )
    X = layers.Dropout(0.3)(X)
    out_2 = layers.LayerNormalization()(out_1 + X)

    X_ffn = layers.Dense(key_dim * 4, activation='gelu')(out_2)
    X_ffn = layers.Dense(key_dim)(X_ffn)
    X_ffn = layers.Dropout(0.3)(X_ffn)
    X = layers.LayerNormalization()(out_2 + X_ffn)

    output_layer = layers.Dense(vocab_size + 2)(X)
    return models.Model(inputs=[input_embeding, input_decoder, encoder_mask], outputs=output_layer)


class TranslationModel(models.Model):
    def __init__(self, encoder: models.Model, decoder: models.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def compile(self, optimizer, loss, metrics=None):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def call(self, input_phrase, decoder_input, training=False):
        embedings, encoder_mask = self.encoder(input_phrase, training=training)
        decoder_output = self.decoder({"encoder_outputs": embedings,
                                       "decoder_input": decoder_input,
                                       "encoder_mask": encoder_mask},
                                      training=training)
        return decoder_output

    def train_step(self, inputs):
        input_phrase, output_phrases = inputs['input_phrase'], inputs['output_phrase']
        decoder_input = output_phrases
        target_output = tf.pad(output_phrases[:, 1:], [[0, 0], [0, 1]])
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
        input_phrase, output_phrases = inputs['input_phrase'], inputs['output_phrase']
        decoder_input = output_phrases
        target_output = tf.pad(output_phrases[:, 1:], [[0, 0], [0, 1]])

        predictions = self(input_phrase, decoder_input, training=False)
        loss = self.compute_loss(y=target_output, y_pred=predictions)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(target_output, predictions)
 
        return {m.name: m.result() for m in self.metrics}
    
    def summary(self, *args, **kwargs):
        self.encoder.summary(*args, **kwargs)
        self.decoder.summary(*args, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder': serialize_keras_object(self.encoder),
            'decoder': serialize_keras_object(self.decoder),
        })
        return config
    
    @classmethod
    def from_config(cls, config, *args, **kwargs):
        encoder = deserialize_keras_object(config['encoder'])
        decoder = deserialize_keras_object(config['decoder'])
        instance = cls(encoder=encoder, decoder=decoder)
        return instance
    
    def save(self, filepath, *args, **kwargs):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        return super().save(filepath, *args, **kwargs)
      

if __name__ == "__main__":
    batch_size = 32
    laungage_couples=[('en', 'pt')]
    num_heads = 8
    key_dim = 256
    vocab_size = 10_000
    text_leight = 64
    model_name = 'simple_en_pt_model'
    tokeniser = MultyLanguageTokenizer(
        f"tokenizer_model_BPE_{vocab_size}", ["en", "pt"], text_leight
    ) 
    data_generator = TranslationDataGenerator(
        laungage_couples, batch_size,
        'train', post_proces=tokeniser.train
    )
    data_generator_val = TranslationDataGenerator(
        laungage_couples, batch_size,
        'validation', post_proces=tokeniser.train
    )

    encoder_model = create_encoder_model(vocab_size, text_leight, num_heads, key_dim)
    decoder_model = crate_decoder_model(vocab_size, text_leight, num_heads, key_dim)

    translation_model = TranslationModel(encoder=encoder_model, decoder=decoder_model)
    loss = losses.SparseCategoricalCrossentropy(
        from_logits=True,
        ignore_class=0
    )
    optimizer = optimizers.Adam(learning_rate=1e-4)
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
        ),
        callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)
    ]
    try:
        translation_model.fit(data_generator,
                              epochs=30,
                              callbacks=callback,
                              validation_data=data_generator_val,
                              # steps_per_epoch=2000,
                              validation_steps=100)
    except KeyboardInterrupt:
        pass
    translation_model.save(f'models/{model_name}/translation_model_final.keras')

