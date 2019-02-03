import pickle

from gensim.models import KeyedVectors

import numpy as np
from numpy.random import random

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class CorpusProcessing:
    def __init__(self, corpus):
        self.corpus = corpus
        self.PADDED = 0
        self.UNKNOWN = 1
        self.max_sent_length = 0
        self.max_word_length = 0

        self.train_tokens = None
        self.train_tags = None
        self.test_tokens = None
        self.test_tags = None
        self.validation_tokens = None
        self.validation_tags = None

        self.char2idx = None
        self.idx2char = None
        self.token2idx = None
        self.idx2token = None
        self.tag2idx = None
        self.idx2tag = None
        self.embeddings = None

        self.folds = 3
        self.train_x_folds = [None] * self.folds
        self.train_y_folds = [None] * self.folds
        self.test_x_folds = [None] * self.folds
        self.test_y_folds = [None] * self.folds

    def read_file(self, full_path):

        if self.corpus == 'comtravo':
            for fold in range(self.folds):
                print("Loading fold {}".format(fold))
                # Loading training
                with open(full_path+'/train_msg_' + str(fold) + '.pkl', 'rb') as f_out:
                    self.train_x_folds[fold] = pickle.load(f_out)
                with open(full_path+'/train_tags_' + str(fold) + '.pkl', 'rb') as f_out:
                    self.train_y_folds[fold] = pickle.load(f_out)

                # Load testing
                with open(full_path+'/test_msg_' + str(fold) + '.pkl', 'rb') as f_out:
                    self.test_x_folds[fold] = pickle.load(f_out)
                with open(full_path+'/test_tags_' + str(fold) + '.pkl', 'rb') as f_out:
                    self.test_y_folds[fold] = pickle.load(f_out)
                fold += 1

        else:
            with open(full_path, 'rt') as f_input:
                sentences_tokens = []
                sentences_tags = []
                tokens = []
                tags = []
                for line in f_input:
                    if self.corpus == 'connl2003' and line.startswith('-DOCSTART-'):
                        continue
                    if line == '\n':
                        if not tags:
                            continue
                        sentences_tokens.append(tokens)
                        sentences_tags.append(tags)
                        tokens = []
                        tags = []
                    else:
                        if self.corpus == 'connl2003':
                            token, pos_tag, chunk_tag, ner_tag = line.split(' ')
                        elif self.corpus == 'paramopama':
                            token, ner_tag = line.split('\t')
                        elif self.corpus == 'cintil':
                            token, ner_tag = line.split('\t')

                        tokens.append(token)
                        tags.append(ner_tag.strip())

            return sentences_tokens, sentences_tags

    def get_max_sent_len(self):
        for sentence in self.train_tokens:
            self.max_sent_length = max(self.max_sent_length, len(sentence))

    def get_max_word_len(self):
        max_token = ''
        for sentence in self.train_tokens:
            for token in sentence:
                if len(token) > self.max_word_length:
                    max_token = token
                    self.max_word_length = len(token)

        print("longest token:")
        print(max_token)
        print()

    def process(self, directory):
        """
        Create an index of tokens, tags, and characters based on the training data and sets
        the maximum sequence length also based on the training data

        :param directory:
        :return:
        """
        if self.corpus == 'connl2003':
            self.train_tokens, self.train_tags = self.read_file(directory + "train.txt")
            self.test_tokens, self.test_tags = self.read_file(directory + "test.txt")
            self.validation_tokens, self.validation_tags = self.read_file(directory + "test.txt")

        elif self.corpus == 'paramopama':
            self.train_tokens, self.train_tags = self.read_file(
                directory + "corpus_paramopama+second_harem.txt")

        elif self.corpus == 'cintil':
            self.train_tokens, self.train_tags = self.read_file(
                directory + "CINTIL-CoNNL_NO_MSC_NO_EVT_fixed.tsv")

        elif self.corpus == 'comtravo':
            self.read_file(directory)

        if self.corpus != 'comtravo':
            # build char, token and tag_index
            self.build_char_index()
            self.build_token_index()
            self.build_tag_index()

            # set the maximum sequence length, and max word length used in padding
            self.get_max_sent_len()
            self.get_max_word_len()

    def build_tag_index(self):
        # index of tags
        all_tags = {tag for sentence_tags in self.train_tags for tag in sentence_tags}
        self.tag2idx = {"PADDED": self.PADDED}
        self.tag2idx.update({tag: i + 1 for i, tag in enumerate(all_tags, 0)})
        self.idx2tag = {value: key for key, value in self.tag2idx.items()}

    def build_token_index(self):
        # index of tokens
        vocabulary = {token for sentence in self.train_tokens for token in sentence}
        self.token2idx = {word: i + 2 for i, word in enumerate(vocabulary, 0)}
        self.token2idx["PADDED"] = self.PADDED
        self.token2idx["UNKNOWN"] = self.UNKNOWN
        self.idx2token = {value: key for key, value in self.token2idx.items()}

    def build_char_index(self):
        # index of chars
        chars = {char for sentence in self.train_tokens for token in sentence for char in token}
        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.char2idx["UNKNOWN"] = self.UNKNOWN
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def vectorize_x_data(self, data, idx):

        # TODO: glove embeddings apparently only contain lowercase words

        unknown_tokens = 0
        vectors = []
        for sentence in data:
            vector = []
            for element in sentence:
                if element in idx:
                    vector.append(idx[element])
                else:
                    unknown_tokens += 1
                    vector.append(self.UNKNOWN)
            vectors.append(vector)

        print("unknown tokens: ", unknown_tokens)

        return pad_sequences(vectors, padding='post', maxlen=self.max_sent_length,
                             truncating='post', value=self.token2idx['PADDED'])

    def vectorize_y_data(self, data):
        vectors = []
        for sentence in data:
            vector = []
            for tag in sentence:
                vector.append(self.tag2idx[tag])
            vectors.append(vector)

        y = pad_sequences(vectors, padding='post', maxlen=self.max_sent_length,
                          truncating='post', value=self.tag2idx['PADDED'])

        print(y.shape)

        return to_categorical(y, len(self.tag2idx.keys())).astype(int)

    def token2char(self, token):
        word_char_idx = []
        for char in token:
            if char in self.char2idx:
                word_char_idx.append(self.char2idx[char])
            else:
                word_char_idx.append(self.char2idx['UNKNOWN'])
        return np.array(word_char_idx)

    def pad_nested_sequences(self, sequences, dtype='int32'):

        """Pads nested sequences to the same length.

        This function transforms a list of list sequences into a 3D Numpy array of shape:

        `(num_samples, max_sent_len, max_word_len)`.

        Args:
            sequences: List of lists of lists.
            dtype: Type of the output sequences.

        # Returns
            x: Numpy array.
        """
        max_sent_len = self.max_sent_length
        max_word_len = self.max_word_length

        """
        for sent in sequences:
            max_sent_len = max(len(sent), max_sent_len)
            for word in sent:
                max_word_len = max(len(word), max_word_len)
        """

        print()
        print("max_sent: ", max_sent_len)
        print("max_word: ", max_word_len)
        print("sequences: ", len(sequences))
        print()

        x = np.zeros((len(sequences), max_sent_len, max_word_len)).astype(dtype)

        for i, sent in enumerate(sequences):
            for j, word in enumerate(sent):
                if j >= max_sent_len:
                    continue
                x[i, j, :len(word)] = word

        return x

    def convert_tags(self):
        """
        Convert tagging schema from IO to BIO
        """
        new_encoded_sents = []
        for sent in self.train_tags:
            new_encoding = []
            for idx, tag in enumerate(sent):
                if tag in ['ORGANIZACAO', 'TEMPO', 'LOCAL', 'PESSOA']:
                    if idx == 0:
                        new_encoding.append('B-' + tag)
                    if self.train_tags[idx - 1] == self.train_tags[idx] and idx > 0:
                        new_encoding.append('I-' + tag)
                    elif self.train_tags[idx - 1] != self.train_tags[idx] and idx > 0:
                        new_encoding.append('B-' + tag)
                else:
                    new_encoding.append(tag)
            new_encoded_sents.append(new_encoding)

        # update train_tags
        self.train_tags = new_encoded_sents

        # updated tags indexes
        self.build_tag_index()

    def split_corpus(self, split=0.8):
        """
        Splits an already processed corpus into train/testing data

        :param split:
        :return:
        """

        split_size = int(split * len(self.train_tokens))

        # test data
        self.test_tokens = self.train_tokens[split_size:]
        self.test_tags = self.train_tags[split_size:]

        # train data
        self.train_tokens = self.train_tokens[0:split_size]
        self.train_tags = self.train_tags[0:split_size]

    def load_embeddings(self, filename, emb_type='word2vec'):
        if emb_type == 'word2vec':

            # input from the original C word2vec-tool format
            #embeddings = KeyedVectors.load_word2vec_format(filename, binary=True)

            # KeyedVectors from gensim
            embeddings = KeyedVectors.load(filename)

            print("embeddings size: ", embeddings.vector_size)

            # create a Keras embedding layer which can plugged-in directly in the model
            self.embeddings = embeddings.get_keras_embedding()

            # updated self.idx2token and token2idx
            self.token2idx = {word: i + 2 for i, word in enumerate(embeddings.index2word)}
            self.token2idx["PADDED"] = self.PADDED
            self.token2idx["UNKNOWN"] = self.UNKNOWN
            self.idx2token = {v: k for v, k in enumerate(embeddings.index2word)}

        elif emb_type == 'glove':

            embeddings_size = 100

            with open('embeddings/GloVe/' + filename) as f_in:
                embeddings = {}
                for line in f_in:
                    splitted_line = line.split()
                    word = splitted_line[0]
                    embedding = np.array([float(val) for val in splitted_line[1:]])
                    embeddings[word] = embedding
                print("Done.", len(embeddings), " words loaded!")

            # updated self.idx2token and token2idx
            self.token2idx = {word: i + 2 for i, word in enumerate(embeddings)}
            self.token2idx["PADDED"] = self.PADDED
            self.token2idx["UNKNOWN"] = self.UNKNOWN
            self.idx2token = {v: k for v, k in enumerate(embeddings)}

            # create a numpy matrix with the embeddings
            embedding_matrix = random((len(self.idx2token)+1, embeddings_size))
            for idx in self.idx2token:
                embedding_matrix[idx] = embeddings[self.idx2token[idx]]
            self.embeddings = embedding_matrix
