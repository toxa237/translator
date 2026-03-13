import numpy as np
import tensorflow as tf
from utils import castom_layers
from simple_model.simple_model import TranslationModel
from tokenizers_modeles.tokenizer_prepr import MultyLanguageTokenizer


def translate_phrase(model, tokenizer, phrase, src_lang, target_lang, max_length=64):
    encoder_input = tokenizer.encode([phrase], [src_lang])
    encoder_input = tf.cast(encoder_input, dtype=tf.int32)

    start_token = 1 
    end_token = 3
    
    decoder_input_array = np.zeros((1, max_length), dtype=np.int32)
    decoder_input_array[0, 0] = start_token
    
    for i in range(max_length - 1):
        predictions = model(encoder_input, decoder_input_array, training=False)
        predicted_id = tf.argmax(predictions[0, i, :], axis=-1).numpy()
        decoder_input_array[0, i + 1] = predicted_id
        if predicted_id == end_token:
            break
            
    result_ids = decoder_input_array[0].tolist()
    translated_text = tokenizer.decode([result_ids], [target_lang])[0]
    
    return translated_text

if __name__ == "__main__":
    vocab_size = 10_000
    text_leight = 64
    model_path = 'models/simple_en_pt_model/translation_model_final.keras'
    
    # Спробуй завантажити модель
    custom_objects = {
        "PositionalEncoding": castom_layers.PositionalEncoding,
        "PadMask": castom_layers.PadMask,
        "DecoderMask": castom_layers.DecoderMask,
        "TranslationModel": TranslationModel 
    }

    tokeniser = MultyLanguageTokenizer(
        f"tokenizer_model_BPE_{vocab_size}", ["en", "pt"], text_leight
    ) 
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    test_phrases = [
        "How are you?",
        "This is a translation test.",
        "The weather is good today."
    ]
    
    for p in test_phrases:
        translation = translate_phrase(model, tokeniser, p, 'en', 'pt')
        print(f"Input: {p}")
        print(f"Translation: {translation}")
        print("-" * 20)
