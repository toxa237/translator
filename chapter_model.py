import tensorflow as tf
from keras import models, layers, optimizers, losses, callbacks
from keras.saving import serialize_keras_object, deserialize_keras_object
from data_base.data_generator import TranslationDataGenerator
from utils.text_vectorise import custom_standardize, vocab_creator
from utils import castom_layers


def create_text_vectorization(vocab):
    indices = layers.TextVectorization(vocabulary=vocab, split="character",
                       standardize=custom_standardize,  # type: ignore
                       output_sequence_length=800, name='text_vectorise')
    return indices


def create_encoder_model(vocab_leigth, num_heads=8, key_dim=256) -> models.Model:
    input_layer = layers.Input(shape=(800,), name='input_phrase')

    X = layers.Embedding(input_dim=vocab_leigth+2, output_dim=key_dim, name='embeding')(input_layer)

    masking_layer = castom_layers.PadMask()(input_layer)

    X = layers.Conv1DTranspose(key_dim*2, 5, activation='tanh')(X)
    X = layers.Conv1D(key_dim, 5, activation='tanh')(X)
    
    X = castom_layers.PositionalEncoding(max_len=800, d_model=key_dim)(X)

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


def crate_decoder_model(vocab_leigth, num_heads=8, key_dim=256) -> models.Model:
    input_embeding = layers.Input(shape=(800, key_dim), name='embeding')
    input_decoder = layers.Input(shape=(799,), name='decoder_input')

    emb_decoder = layers.Embedding(input_dim=vocab_leigth+2, output_dim=key_dim, name='emb_decoder')(input_decoder)
    
    emb_decoder = layers.ZeroPadding1D(padding = (8, 0))(emb_decoder) # type: ignore
    emb_decoder = layers.Reshape((807, key_dim, 1))(emb_decoder)
    emb_decoder = layers.Conv2D(key_dim*2, (5, key_dim), activation='tanh')(emb_decoder)
    emb_decoder = layers.Reshape((803, key_dim*2, 1))(emb_decoder)
    emb_decoder = layers.Conv2D(key_dim, (5, key_dim*2), activation='tanh')(emb_decoder)
    emb_decoder = layers.Reshape((799, key_dim))(emb_decoder)
    
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
    output_layer = layers.Dense(vocab_leigth + 2)(output_layer)
    
    model = models.Model(inputs=[input_embeding, input_decoder], outputs=output_layer)
    return model


class TranslationModel(models.Model):
    def __init__(self, encoder: models.Model, decoder: models.Model,
                 text_vector: layers.TextVectorization, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.text_vector = text_vector
    
    def compile(self, optimizer, loss, metrics=None):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def call(self, input_phrase, decoder_input, training=False):
        embedings = self.encoder(input_phrase, training=training)
        decoder_output = self.decoder([embedings, decoder_input], training=training)
        return decoder_output

    def train_step(self, inputs):
        input_phrase, output_phrases = inputs['input_phrase'], inputs['decoder_input']
        input_phrase = self.text_vector(input_phrase)
        output_phrases = self.text_vector(output_phrases)
        decoder_input = output_phrases[:, :-1]
        target_output = output_phrases[:, 1:]
        # decoder_input = output_phrases
        # target_output = tf.pad(output_phrases[:, 1:], [[0, 0], [0, 1]], constant_values=0)
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
        input_phrase = self.text_vector(input_phrase)
        output_phrases = self.text_vector(output_phrases)

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
    
    def summary(self, *args, **kwargs):
        self.encoder.summary(*args, **kwargs)
        self.decoder.summary(*args, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder': serialize_keras_object(self.encoder),
            'decoder': serialize_keras_object(self.decoder),
            'text_vector': serialize_keras_object(self.text_vector),
        })
        return config
    
    @classmethod
    def from_config(cls, config, *args, **kwargs):
        encoder = deserialize_keras_object(config['encoder'])
        decoder = deserialize_keras_object(config['decoder'])
        text_vector= deserialize_keras_object(config['text_vector'])
        instance = cls(encoder=encoder, decoder=decoder, text_vector=text_vector)
        return instance
       
    def predict_with_decode(self, input_data):
        input_data = self.predict(input_data)
        token_ids = tf.argmax(input_data, axis=-1)
        vocab = self.text_vector.get_vocabulary()
        decoded_texts = []
        for row in token_ids:
            chars = [vocab[i] for i in row if vocab[i] not in ['<PAD>', '<END>']]
            decoded_texts.append("".join(chars))

        return decoded_texts


if __name__ == "__main__":
    batch_size = 8
    laungage_couples=[('en', 'pt')]
    num_heads = 8
    key_dim = 256
    model_name = 'model_t'
    data_generator = TranslationDataGenerator(laungage_couples, batch_size, 'train')
    data_generator_val = TranslationDataGenerator(laungage_couples, batch_size, 'validation')
    vocab = vocab_creator(data_generator.unique_language_list)

    text_vector = create_text_vectorization(vocab) 
    encoder_model = create_encoder_model(len(vocab), num_heads, key_dim)
    decoder_model = crate_decoder_model(len(vocab), num_heads, key_dim)
    
    translation_model = TranslationModel(encoder=encoder_model, decoder=decoder_model,
                                         text_vector=text_vector)
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

