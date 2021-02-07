import pickle

import numpy as np
from nltk.stem.snowball import RomanianStemmer
from nltk.tokenize import word_tokenize
from tensorflow import keras

SENT_SIZE = 100
CHAR_FEAT_SIZE = 10


def title_and_text_model(title_char2idx, text_char2idx):
    # TITLE INPUT
    title_input = keras.layers.Input(shape=(SENT_SIZE, CHAR_FEAT_SIZE))
    title_char_emb = keras.layers.TimeDistributed(keras.layers.Embedding(input_dim=len(title_char2idx) + 1,
                                                                         output_dim=30, input_length=CHAR_FEAT_SIZE))(
        title_input)
    title_char_dropout = keras.layers.Dropout(0.5)(title_char_emb)
    title_char_conv1d = keras.layers.TimeDistributed(keras.layers.Conv1D(kernel_size=3, filters=32,
                                                                         padding='same', activation='tanh', strides=1))(
        title_char_dropout)
    title_char_maxpool = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(CHAR_FEAT_SIZE))(title_char_conv1d)
    title_char_feats = keras.layers.TimeDistributed(keras.layers.Flatten())(title_char_maxpool)

    # TEXT INPUT
    text_input = keras.layers.Input(shape=(SENT_SIZE, CHAR_FEAT_SIZE))
    text_char_emb = keras.layers.TimeDistributed(keras.layers.Embedding(input_dim=len(text_char2idx) + 1,
                                                                        output_dim=30, input_length=CHAR_FEAT_SIZE))(
        text_input)
    text_char_dropout = keras.layers.Dropout(0.5)(text_char_emb)
    text_char_conv1d = keras.layers.TimeDistributed(keras.layers.Conv1D(kernel_size=3, filters=32,
                                                                        padding='same', activation='tanh', strides=1))(
        text_char_dropout)
    text_char_maxpool = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(CHAR_FEAT_SIZE))(text_char_conv1d)
    text_char_feats = keras.layers.TimeDistributed(keras.layers.Flatten())(text_char_maxpool)
    all_feat = keras.layers.concatenate([title_char_feats, text_char_feats])
    all_out = keras.layers.SpatialDropout1D(0.3)(all_feat)
    bi_lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=100,
                                                           return_sequences=False))(all_out)

    out = keras.layers.Dense(1, activation="sigmoid")(bi_lstm)

    model = keras.models.Model([title_input, text_input], out)
    model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['acc'])

    return model


def process(txt, char2idx):
    def clean(txt):
        tokens = word_tokenize(txt)
        stemmer = RomanianStemmer()
        # remove all tokens that are not alphabetic
        words = [stemmer.stem(word.lower()) for word in tokens if word.isalpha()]
        return words

    def get_char_features(X, char2idx, sent_size=100, char_feat_size=10):
        def word2charidxs(word, char2idx):
            char_feats = list(map(lambda c: char2idx.get(c, 0), word))
            return char_feats

        X_chars = []
        for sent in X:
            sent_indx = list(map(lambda x: word2charidxs(x, char2idx), sent))
            sent_indx = keras.preprocessing.sequence.pad_sequences(maxlen=char_feat_size,
                                                                   sequences=sent_indx, padding="post",
                                                                   truncating="post", value=0)
            X_chars.append(sent_indx)
        pad_val = np.zeros((sent_size, char_feat_size))
        X_chars = keras.preprocessing.sequence.pad_sequences(maxlen=sent_size, sequences=X_chars,
                                                             padding="post", truncating="post", value=pad_val)
        return X_chars

    words = clean(txt)

    return get_char_features([words], char2idx)[0]


def load_model(weights_path='model_weights/model.h5', char2idx_path='model_weights/char2idxs.pkl'):
    with open(char2idx_path, 'rb') as f:
        char2idxs = pickle.load(f)
    model = title_and_text_model(char2idxs['title_char2idx'], char2idxs['text_char2idx'])
    print(model.summary())

    model.load_weights(weights_path)
    return model, char2idxs
