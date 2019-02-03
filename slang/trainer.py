import numpy as np

from keras.callbacks import Callback
from sklearn_crfsuite.metrics import flat_classification_report

from slang.models import NeuralArchitectureForNER
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


class F1_score(Callback):
    def __init__(self, x_true, y_true, corpus):
        super(F1_score, self).__init__()
        self.x_true = x_true
        self.y_true = y_true
        self.corpus = corpus

    # make prediction on validation data
    def on_epoch_end(self, epoch, logs={}):
        all_pred_labels = []
        all_true_labels = []
        y_pred = self.model.predict(self.x_true)

        for sent_pred, sent_true in zip(y_pred, self.y_true):
            pred = [self.corpus.idx2tag[np.argmax(word)] for word in sent_pred]
            true = [self.corpus.idx2tag[np.argmax(word)] for word in sent_true]

            all_true_labels.extend(true)
            all_pred_labels.extend(pred)

        score = f1_score(all_true_labels, all_pred_labels, average='weighted')
        print(' - f1: {:04.2f}'.format(score * 100))
        print(classification_report(all_true_labels, all_pred_labels))
        logs['f1'] = score


class Trainer:
    def __init__(self, corpus):
        self.corpus = corpus
        pass

    def evaluate(self, predictions):
        all_predictions = []
        for pred_sent_tags in predictions:
            sentence_predictions = []
            for tag in pred_sent_tags:
                sentence_predictions.append(self.corpus.idx2tag[np.argmax(tag)])
            all_predictions.append(sentence_predictions)
        all_true = []

        for sentence in self.corpus.vectorize_y_data(self.corpus.test_tags):
            sent_true = [self.corpus.idx2tag[np.argmax(tag)] for tag in sentence]
            all_true.append(sent_true)

        labels = set(self.corpus.tag2idx.keys())
        labels.remove('PADDED')
        if self.corpus.corpus == 'comtravo':
            labels.remove('O-')
        else:
            labels.remove('O')

        print(flat_classification_report(all_true, all_predictions, labels=sorted(list(labels))))

    def train_and_evaluate(self, epochs=2, lr=0.01):

        # vectorize and pad train data
        print("\nVectorizing training data")
        x_words = self.corpus.vectorize_x_data(self.corpus.train_tokens, self.corpus.token2idx)
        y_tags = self.corpus.vectorize_y_data(self.corpus.train_tags)
        x_char = [[self.corpus.token2char(token) for token in sent]
                  for sent in self.corpus.train_tokens]
        x_char_padded = self.corpus.pad_nested_sequences(x_char)

        # take some portion of the training data as validation
        validation_split = 0.10  # last 10% of training data
        split_index = len(x_words)-int(len(x_words) * validation_split)

        print("train data : ", len(x_words))
        print("split_index: ", split_index)

        print()
        print("words: ", x_words.shape)
        print("tags : ", y_tags.shape)
        print("chars: ", x_char_padded.shape)
        print()

        x_train_words = x_words[0:split_index, ]
        y_train_tags = y_tags[0:split_index, ]
        x_train_char_padded = x_char_padded[0:split_index, ]

        x_validation_words = x_words[split_index:, ]
        y_validation_tags = y_tags[split_index:, ]
        x_validation_padded = x_char_padded[split_index:, ]

        print()
        print("words: ", x_train_words.shape)
        print("tags : ", y_train_tags.shape)
        print("chars: ", x_train_char_padded.shape)
        print()
        print("words: ", x_validation_words.shape)
        print("tags : ", y_validation_tags.shape)
        print("chars: ", x_validation_padded.shape)
        print()

        # vectorize and pad test data
        print("\nVectorizing testing data")
        x_test_words = self.corpus.vectorize_x_data(self.corpus.test_tokens, self.corpus.token2idx)
        x_test_char = [[self.corpus.token2char(token) for token in sent]
                       for sent in self.corpus.test_tokens]

        x_test_char_padded = self.corpus.pad_nested_sequences(x_test_char)

        # set the model parameters
        vocab_size = len(self.corpus.idx2token)
        num_labels = len(self.corpus.idx2tag)
        use_crf = True
        use_char = True
        char_vocab_size = len(self.corpus.char2idx)
        char_embeddings_dim = 25

        bilstm = NeuralArchitectureForNER(self.corpus.max_sent_length, vocab_size, num_labels,
                                          char_vocab_size, char_embeddings_dim,
                                          embeddings=self.corpus.embeddings, word_lstm_units=64,
                                          use_crf=use_crf, use_char=use_char)
        model = bilstm.get_model()

        # ToDo: optionally save plots from each epoch to files

        if use_char:
            print("\nTraining data:")
            print("words: ", x_train_words.shape)
            print("chars: ", x_train_char_padded.shape)
            print("tags : ", y_train_tags.shape)
            print()
            print("Test data:")
            print("words: ", x_test_words.shape)
            print("chars: ", x_test_char_padded.shape)
            print()
            history = model.fit([x_train_words, x_train_char_padded], y_train_tags,
                                batch_size=32, epochs=epochs)
            predictions = model.predict([x_test_words, x_test_char_padded])

        else:

            f1_score = F1_score(x_validation_words, y_validation_tags, self.corpus)

            print("\nTraining data:")
            print("words: ", x_train_words.shape)
            print()
            print("Test data:")
            print("words: ", x_test_words.shape)
            print()

            history = model.fit(x_train_words, y_train_tags, batch_size=64,
                                epochs=epochs,
                                validation_data=[x_validation_words, y_validation_tags],
                                callbacks=[f1_score])

            print("Applying model on test data")
            predictions = model.predict(x_test_words)

        print("\nmax_seq_len   : ", bilstm.max_seq_len)
        print("vocab_size    : ", bilstm.word_vocab_size)
        print("num_labels    : ", bilstm.num_labels)
        print("embeddings dim: ", bilstm.word_embeddings_dim)
        print("use_crf       : ", bilstm.use_crf)
        print("use_char      : ", bilstm.use_char)
        print("lstm_units    : ", bilstm.word_lstm_units)
        print()

        bilstm.save_model()

        self.evaluate(predictions)

    def train_test_fold(self, train_x, train_y, test_x, test_y, epochs):

        self.corpus.train_tokens = train_x
        self.corpus.train_tags = train_y

        self.corpus.test_tokens = test_x
        self.corpus.test_tags = test_y

        # build char, token and tag_index
        self.corpus.build_char_index()
        self.corpus.build_token_index()
        self.corpus.build_tag_index()

        # set the maximum sequence length, and max word length used in padding
        self.corpus.get_max_sent_len()
        self.corpus.get_max_word_len()

        print("max_sent_lenght: ", self.corpus.max_sent_length)
        print("max_word_lenght: ", self.corpus.max_word_length)

        # vectorize and pad train data
        print("\nVectorizing training data")
        x_train_words = self.corpus.vectorize_x_data(train_x, self.corpus.token2idx)
        y_train_tags = self.corpus.vectorize_y_data(train_y)
        # x_train_char = [[self.corpus.token2char(token) for token in sent] for sent in train_x]
        # x_train_char_padded = self.corpus.pad_nested_sequences(x_train_char)

        # vectorize and pad test data
        print("\nVectorizing testing data")
        x_test_words = self.corpus.vectorize_x_data(test_x, self.corpus.token2idx)
        # x_test_char = [[self.corpus.token2char(token) for token in sent] for sent in train_x]
        # x_test_char_padded = self.corpus.pad_nested_sequences(x_test_char)

        # set the model parameters
        vocab_size = len(self.corpus.idx2token)
        num_labels = len(self.corpus.idx2tag)
        use_crf = True
        use_char = False
        char_vocab_size = len(self.corpus.char2idx)
        char_embeddings_dim = 25

        bilstm = NeuralArchitectureForNER(self.corpus.max_sent_length, vocab_size, num_labels,
                                          char_vocab_size, char_embeddings_dim,
                                          embeddings=self.corpus.embeddings, word_lstm_units=64,
                                          use_crf=use_crf, use_char=use_char)
        model = bilstm.get_model()

        if use_char:
            print("\nTraining data:")
            print("words: ", x_train_words.shape)
            print("chars: ", x_train_char_padded.shape)
            print("tags : ", y_train_tags.shape)
            print()
            print("Test data:")
            print("words: ", x_test_words.shape)
            print("chars: ", x_test_char_padded.shape)
            print()
            # ToDo: add precision/recall/f1 during training
            # ToDo: add plots for each epoch
            history = model.fit([x_train_words, x_train_char_padded], y_train_tags,
                                batch_size=32, epochs=epochs)
            predictions = model.predict([x_test_words, x_test_char_padded])

        else:
            print("\nTraining data:")
            print("words: ", x_train_words.shape)
            print()
            print("Test data:")
            print("words: ", x_test_words.shape)
            print()
            history = model.fit(x_train_words, y_train_tags, batch_size=32, epochs=epochs)

            print("Applying model on test data")
            predictions = model.predict(x_test_words)

        print("\nmax_seq_len   : ", bilstm.max_seq_len)
        print("vocab_size    : ", bilstm.word_vocab_size)
        print("num_labels    : ", bilstm.num_labels)
        print("embeddings dim: ", bilstm.word_embeddings_dim)
        print("use_crf       : ", bilstm.use_crf)
        print("use_char      : ", bilstm.use_char)
        print("lstm_units    : ", bilstm.word_lstm_units)
        print()

        bilstm.save_model()

        self.evaluate(predictions, self.corpus)

    def cross_fold_evaluation(self, epochs):

        for i in range(3):
            print("fold {}".format(i))
            print("train x: ", len(self.corpus.train_x_folds[i]))
            print("train y: ", len(self.corpus.train_y_folds[i]))
            print("test x: ", len(self.corpus.test_x_folds[i]))
            print("test y: ", len(self.corpus.test_y_folds[i]))

            train_x = self.corpus.train_x_folds[i]
            train_y = self.corpus.train_y_folds[i]

            test_x = self.corpus.test_x_folds[i]
            test_y = self.corpus.test_y_folds[i]

            self.train_test_fold(train_x, train_y, test_x, test_y, epochs)
