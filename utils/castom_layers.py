from keras import ops
from keras.layers import Embedding, Layer, serialize, deserialize
import tensorflow as tf


class PadMask(Layer):
    def __init__(self, pad_index=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_index = pad_index
    
    def build(self, input_shape):  # type: ignore
        super().build(input_shape)
        return [input_shape[0], 1, 1, *input_shape[1:]]

    def call(self, input_data):
        masking_layer = tf.not_equal(input_data, self.pad_index)
        masking_layer = tf.cast(masking_layer, tf.float32)
        return masking_layer[:, tf.newaxis, tf.newaxis, :]  # type: ignore

    def get_config(self):
        return {"pad_index": self.pad_index}

    @classmethod
    def from_confige(cls, confige):
        return cls(**confige)


class DecoderMask(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_mask = PadMask()

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, x):
        batch_size, seq_len = ops.shape(x)[0], ops.shape(x)[1]

        i = tf.range(seq_len)[:, tf.newaxis]
        j = tf.range(seq_len)
        lookahead_mask = tf.cast(i >= j, tf.float32)
        lookahead_mask = tf.reshape(lookahead_mask, [1, seq_len, seq_len])

        pad_mask = self.pad_mask(x) 

        lookahead_mask = tf.tile(lookahead_mask, [batch_size, 1, 1])
        lookahead_mask = tf.expand_dims(lookahead_mask, axis=1)

        combined_mask = tf.minimum(pad_mask, lookahead_mask)
        return combined_mask


class PositionalEncoding(Layer):
    def __init__(self, max_len=None, d_model=None, pos_emb=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(pos_emb, Embedding):
            self.pos_emb = pos_emb
        else:
            self.pos_emb = Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1] # type: ignore
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        return x + positions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "pos_emb": serialize(self.pos_emb)
        })
        return config

    @classmethod
    def from_config(cls, config):
        pos_emb = config.get("pos_emb", None)
        if pos_emb:
            config["pos_emb"] = deserialize(config["pos_emb"])
        return cls(**config)




