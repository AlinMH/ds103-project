from tensorflow import keras
import pickle

TITLE_SENT_SIZE = 10
TEXT_SENT_SIZE = 100
CHAR_FEAT_SIZE = 10

def title_and_text_model(title_char2idx, text_char2idx):
    # TITLE INPUT
    title_input = keras.layers.Input(shape=(TITLE_SENT_SIZE, CHAR_FEAT_SIZE))
    title_char_emb = keras.layers.TimeDistributed(keras.layers.Embedding(input_dim=len(title_char2idx) + 1,
        output_dim=30, input_length=CHAR_FEAT_SIZE))(title_input)  

    title_char_dropout = keras.layers.Dropout(0.5)(title_char_emb)
    title_char_conv1d = keras.layers.TimeDistributed(keras.layers.Conv1D(kernel_size=3, filters=32,
        padding='same',activation='tanh', strides=1))(title_char_dropout)
    title_char_maxpool = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(CHAR_FEAT_SIZE))(title_char_conv1d)
    title_char_feats = keras.layers.TimeDistributed(keras.layers.Flatten())(title_char_maxpool)
    
    # TEXT INPUT
    text_input = keras.layers.Input(shape=(TEXT_SENT_SIZE, CHAR_FEAT_SIZE))
    text_char_emb = keras.layers.TimeDistributed(keras.layers.Embedding(input_dim=len(text_char2idx) + 1,
        output_dim=30, input_length=CHAR_FEAT_SIZE))(text_input)  

    text_char_dropout = keras.layers.Dropout(0.5)(text_char_emb)
    text_char_conv1d = keras.layers.TimeDistributed(keras.layers.Conv1D(kernel_size=3, filters=32,
        padding='same',activation='tanh', strides=1))(text_char_dropout)
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

def load_model(weights_path='../model.h5', char2idx_path='../char2idxs.pkl'):
    with open(char2idx_path, 'rb') as f:
        char2idxs = pickle.load(f)
    model = title_and_text_model(char2idxs['title_char2idx'], char2idxs['text_char2idx'])
    model.load_weights(weights_path)
    return model

if __name__ == '__main__':
    model = load_model()