import tensorflow as tf
from keras.layers import Layer
from keras import ops


class PadMask(Layer):
    def __init__(self, pad_index=0):
        super().__init__()
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
    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, input_data):
        input_shape = ops.shape(input_data)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = ops.arange(sequence_length)[:, None]
        j = ops.arange(sequence_length)
        mask = ops.cast(i >= j, dtype="int32")  # type: ignore
        mask = ops.reshape(mask, (1, sequence_length, sequence_length))
        mult = ops.concatenate(
            [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])],
            axis=0,
        )
        return ops.tile(mask, mult)

