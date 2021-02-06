import nltk
import numpy as np
import tensorflow as tf
from nltk.tokenize import sent_tokenize
from tensorflow.keras import backend as K
from transformers import BertTokenizer, TFBertModel

NB_OF_SENTS = 30

nltk.download("punkt")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = TFBertModel.from_pretrained("bert-base-uncased")


def get_model(lstm_cell_size=150):
    in_id = tf.keras.layers.Input((NB_OF_SENTS, 768), name="input_shape")
    lstm_later, forward_h, forward_c = tf.keras.layers.LSTM(lstm_cell_size, return_sequences=True, return_state=True)(
        in_id)
    linear = tf.keras.layers.Dense(lstm_cell_size)(forward_h)
    attention = tf.keras.layers.dot([lstm_later, linear], axes=(-1))
    attention = tf.keras.layers.Activation('softmax', name='attention_vec')(attention)
    attention = tf.keras.layers.RepeatVector(lstm_cell_size)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)
    sent_representation = tf.keras.layers.multiply([lstm_later, attention])
    sent_representation = tf.keras.layers.Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    sent_representation_final = tf.keras.layers.Concatenate()([sent_representation, forward_h])
    drop = tf.keras.layers.Dropout(0.2)(sent_representation_final)
    predictions = tf.keras.layers.Dense(2, activation='softmax')(drop)
    model = tf.keras.Model(inputs=in_id, outputs=predictions)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model


def get_embedding(text: str):
    text = text.lower()
    sents = sent_tokenize(text)
    snt_good = []
    for idx, snt in enumerate(sents):
        if idx == NB_OF_SENTS:
            break
        snt_good.append(snt)
    while len(snt_good) != NB_OF_SENTS:
        snt_good.append('')

    assert (len(snt_good) == NB_OF_SENTS)
    encoded_inputs = tokenizer(snt_good, padding=True, truncation=True, return_tensors="tf", max_length=60)
    output = bert(encoded_inputs)
    y = tf.keras.layers.GlobalAveragePooling1D()(output[0])
    y = np.reshape(y, (-1, 30, 768))
    return y


def load_model():
    model = get_model()
    model.load_weights('model_weights/')
    return model
