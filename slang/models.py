import json
from datetime import datetime

import numpy
from numpy.random import random

from keras import Input, Model, optimizers
from keras.layers import Embedding, LSTM, Dense, Bidirectional, TimeDistributed, Concatenate
from keras_contrib.layers import CRF
from keras.models import model_from_json


# ToDO: save/load like in anago

def save_model(model, weights_file, params_file):
    with open(params_file, 'w') as f:
        params = model.to_json()
        json.dump(json.loads(params), f, sort_keys=True, indent=4)
        model.save_weights(weights_file)


def load_model(weights_file, params_file):
    with open(params_file) as f:
        model = model_from_json(f.read(), custom_objects={'CRF': CRF})
        model.load_weights(weights_file)

    return model


class NeuralArchitectureForNER:
    """
    Neural Architectures for Named Entity Recognition (Lample et al. 2016)
    http://www.aclweb.org/anthology/N16-1030
    """

    def __init__(self, max_seq_len, word_vocab_size, num_labels, char_vocab_size,
                 char_embeddings_dim, word_embeddings_dim=200, embeddings=None,
                 word_lstm_units=256, char_lstm_units=25, use_crf=True, use_char=False):

        self.max_seq_len = max_seq_len
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.num_labels = num_labels
        self.word_embeddings_dim = word_embeddings_dim
        self.char_embeddings_dim = char_embeddings_dim
        self.embeddings = embeddings
        self.word_lstm_units = word_lstm_units
        self.char_lstm_units = char_lstm_units
        self.use_crf = use_crf
        self.use_char = use_char
        self.model = None

    def get_model(self):
        # ToDo: add dropout
        """
        - we use dropout training (Hinton et al., 2012), applying a dropout mask to the final
          embedding layer just before the input to the bidirectional LSTM,

        - We set the dropout rate to 0.5. Using higher rates negatively impacted our results,
          while smaller rates led to longer training time.

        - learning rate 0.01
        - gradient clipping of 5.0.

        - single layer for the forward and backward LSTMs whose dimensions are set to 100.
          Tuning this dimension did not significantly impact model performance.



        :return:
        """
        word_x = Input(batch_shape=(None, None), dtype='int32', name='word_input')

        if self.embeddings is None:
            embedding_matrix = random((self.word_vocab_size + 1, self.word_embeddings_dim))
            word_embeddings = Embedding(input_dim=self.word_vocab_size + 1,
                                        output_dim=self.word_embeddings_dim,
                                        trainable=True,
                                        mask_zero=True,
                                        weights=[embedding_matrix],
                                        name='word_embedding')(word_x)

        elif isinstance(self.embeddings, numpy.ndarray):
            vocab_size = self.embeddings.shape[0]
            dim = self.embeddings.shape[1]
            word_embeddings = Embedding(input_dim=vocab_size,
                                        output_dim=dim,
                                        trainable=True,
                                        mask_zero=True,
                                        weights=[self.embeddings],
                                        name='word_embedding')(word_x)

        else:
            # ToDo: add some test here

            print()
            print("vocab_size    : ", self.embeddings.input_dim)
            print("embeddings dim: ", self.embeddings.output_dim)

            word_embeddings = self.embeddings(word_x)

        if self.use_char:
            # Character Embeddings
            char_x = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
            char_embeddings = Embedding(input_dim=self.char_vocab_size,
                                        output_dim=self.char_embeddings_dim,
                                        trainable=True,
                                        mask_zero=True,
                                        name='char_embedding')(char_x)

            # Apply a bi-LSTM to each char_embedding
            char_embeddings = TimeDistributed(
                Bidirectional(LSTM(units=self.char_lstm_units,
                                   return_sequences=False,
                                   name='char_lstm')))(char_embeddings)

            # Concatenate all the vectors
            embeddings = Concatenate()([word_embeddings, char_embeddings])
        else:
            embeddings = word_embeddings

        lstm_out = Bidirectional(LSTM(units=self.word_lstm_units, return_sequences=True),
                                 merge_mode='concat',
                                 name='word_lstm')(embeddings)
        dense_out = Dense(100, activation='tanh', name='dense_layer')(lstm_out)

        if self.use_crf:
            crf = CRF(self.num_labels, learn_mode='join', sparse_target=False, name='crf')
            pred = crf(dense_out)
            loss_fun = crf.loss_function
            metrics_fun = crf.accuracy

        else:
            pred = Dense(self.num_labels, activation='softmax', name='final_dense')(dense_out)
            loss_fun = 'categorical_crossentropy'
            metrics_fun = 'accuracy'

        if self.use_char:
            model = Model([word_x, char_x], pred)
        else:
            model = Model(word_x, pred)

        self.model = model

        # ToDo: add another optimizer
        # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.1, nesterov=True)
        model.compile(optimizer="adam", loss=loss_fun, metrics=[metrics_fun])

        return model

    def get_transition_scores(self):
        # ToDo:
        print(self.model.layers)
        pass

    def save_model(self):
        #datetime_now = f"{datetime.now():%Y-%m-%d_%H:%M:%S}"
        #self.model.save('model')
        #save_load_utils.save_all_weights(self.model, "model_weights")
        pass

    def load_model(self):
        # ToDo:
        # save_load_utils.load_all_weights('test_crf_model')
        pass


class biLSTM_CNN_CRF:
    """
    An implementations of the paper:

    "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF" (Xuezhe Ma and Eduard Hovy)
    """

    # ToDo:
    # adicionar mais modelos:
    # https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs

    def __init__(self, max_seq_len, word_vocab_size, num_labels, char_vocab_size,
                 char_embeddings_dim, word_embeddings_dim=200, embeddings=None,
                 word_lstm_units=128, char_lstm_units=25, use_crf=True, use_char=False):

        self.max_seq_len = max_seq_len
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.num_labels = num_labels
        self.word_embeddings_dim = word_embeddings_dim
        self.char_embeddings_dim = char_embeddings_dim
        self.embeddings = embeddings
        self.word_lstm_units = word_lstm_units
        self.char_lstm_units = char_lstm_units
        self.use_crf = use_crf
        self.use_char = use_char
        self.model = None

    def get_model(self):

        word_x = Input(batch_shape=(None, None), dtype='int32', name='word_input')

        if self.embeddings is None:
            embedding_matrix = random((self.word_vocab_size + 1, self.word_embeddings_dim))
            word_embeddings = Embedding(input_dim=self.word_vocab_size + 1,
                                        output_dim=self.word_embeddings_dim,
                                        trainable=True,
                                        mask_zero=True,
                                        weights=[embedding_matrix],
                                        name='word_embedding')(word_x)

        if self.use_char:
            # Character Embeddings
            char_x = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
            char_embeddings = Embedding(input_dim=self.char_vocab_size,
                                        output_dim=self.char_embeddings_dim,
                                        trainable=True,
                                        mask_zero=True,
                                        name='char_embedding')(char_x)

            # Apply a bi-LSTM to each char_embedding
            char_embeddings = TimeDistributed(
                Bidirectional(LSTM(units=self.char_lstm_units,
                                   return_sequences=False,
                                   name='char_lstm')))(char_embeddings)

            # Concatenate all the vectors
            embeddings = Concatenate()([word_embeddings, char_embeddings])
        else:
            embeddings = word_embeddings

        lstm_out = Bidirectional(LSTM(units=self.word_lstm_units, return_sequences=True),
                                 merge_mode='concat',
                                 name='word_lstm')(embeddings)
        dense_out = Dense(100, activation='tanh', name='dense_layer')(lstm_out)

        if self.use_crf:
            crf = CRF(self.num_labels, learn_mode='join', sparse_target=False, name='crf')
            pred = crf(dense_out)
            loss_fun = crf.loss_function
            metrics_fun = crf.accuracy

        else:
            pred = Dense(self.num_labels, activation='softmax', name='final_dense')(dense_out)
            loss_fun = 'categorical_crossentropy'
            metrics_fun = 'accuracy'

        if self.use_char:
            model = Model([word_x, char_x], pred)
        else:
            model = Model(word_x, pred)

        self.model = model

        # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.1, nesterov=True)
        model.compile(optimizer="adam", loss=loss_fun, metrics=[metrics_fun])

        return model