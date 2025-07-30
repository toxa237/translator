import tensorflow as tf
from simple_model import TranslationModel
from utils import castom_layers
from utils.text_vectorise import custom_standardize
from keras.models import load_model

if __name__ == '__main__':
    # Load the model
    model: TranslationModel = load_model('models/translation_model_epoch_03_loss_0.0000.keras',
                                        custom_objects={
                                            'DecoderMask': castom_layers.DecoderMask,
                                            'PadMask': castom_layers.PadMask,
                                            'custom_standardize': custom_standardize,
                                            'TranslationModel': TranslationModel
                                        }
                                    )  # type: ignore

    # Print the model summary
    model.summary()

    # Test the model with a sample input
    sample_input = tf.constant(['Hello', 'my name is', 'where'], dtype=tf.string)
    output = model.encoder.predict(sample_input)
    # print("Encoder output shape:", output.shape)
    output = model.predict_with_decode(sample_input)
    print("Model output:", output)

