import numpy as np
np.random.seed(1337)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Reshape, Flatten, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.layers import concatenate, Input
from keras.models import Model
# graph model has been removed
from keras.preprocessing import sequence


class CNN_module():
    '''
    classdocs
    '''
    batch_size = 128
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, init_W=None):

        self.max_len = max_len
        max_features = vocab_size
        vanila_dimension = 200
        projection_dimension = output_dimesion

        # self.qual_model = Model()
        self.qual_conv_set = {}

        # my code
        '''Input'''
        input_data = Input(shape=(max_len, ), name='input', dtype='int32')

        '''Embedding Layer'''
        if init_W is None:
            embedding_layer = Embedding(max_features, emb_dim, input_length=max_len)(input_data)
        else:
            embedding_layer = Embedding(max_features, emb_dim, weights=[init_W / 20], input_length=max_len)(input_data)

        '''Convolution Layer & Max Pooling Layer'''
        reshape_layer = Reshape((self.max_len, emb_dim, 1))(embedding_layer)

        conv_1 = Conv2D(nb_filters, (3, emb_dim), activation='relu')(reshape_layer)
        maxpool_1 = MaxPooling2D(pool_size=(self.max_len-3+1, 1))(conv_1)
        conv_2 = Conv2D(nb_filters, (4, emb_dim), activation='relu')(reshape_layer)
        maxpool_2 = MaxPooling2D(pool_size=(self.max_len - 4 + 1, 1))(conv_2)
        conv_3 = Conv2D(nb_filters, (5, emb_dim), activation='relu')(reshape_layer)
        maxpool_3 = MaxPooling2D(pool_size=(self.max_len - 5 + 1, 1))(conv_3)

        multiple_layer = concatenate([maxpool_1, maxpool_2, maxpool_3])
        flatten_layer = Flatten()(multiple_layer)

        '''Dropout Layer'''
        layer = Dense(vanila_dimension, activation='tanh')(flatten_layer)
        layer = Dropout(dropout_rate)(layer)

        '''Projection Layer & Output Layer'''
        output_layer = Dense(projection_dimension, activation='tanh')(layer)

        '''Model'''
        self.model = Model(inputs=input_data, outputs=output_layer)

        print("Jay::Model has been completed")

        '''compile'''
        self.model.compile(optimizer='rmsprop',
                           loss='mse')

        print("Jay::compile completed")

        # '''Embedding Layer'''
        # self.model.add_input(name='input', input_shape=(max_len,), dtype=int)
        #
        # if init_W is None:
        #     self.model.add_node(Embedding(
        #         max_features, emb_dim, input_length=max_len), name='sentence_embeddings', input='input')
        # else:
        #     self.model.add_node(Embedding(max_features, emb_dim, input_length=max_len, weights=[
        #                         init_W / 20]), name='sentence_embeddings', input='input')
        #
        # '''Convolution Layer & Max Pooling Layer'''
        # for i in filter_lengths:
        #     model_internal = Sequential()
        #     model_internal.add(
        #         Reshape(dims=(1, self.max_len, emb_dim), input_shape=(self.max_len, emb_dim)))
        #     model_internal.add(Convolution2D(
        #         nb_filters, i, emb_dim, activation="relu"))
        #     model_internal.add(MaxPooling2D(
        #         pool_size=(self.max_len - i + 1, 1)))
        #     model_internal.add(Flatten())
        #
        #     self.model.add_node(model_internal, name='unit_' +
        #                         str(i), input='sentence_embeddings')
        #
        # '''Dropout Layer'''
        # self.model.add_node(Dense(vanila_dimension, activation='tanh'),
        #                     name='fully_connect', inputs=['unit_' + str(i) for i in filter_lengths])
        # self.model.add_node(Dropout(dropout_rate),
        #                     name='dropout', input='fully_connect')
        # '''Projection Layer & Output Layer'''
        # self.model.add_node(Dense(projection_dimension, activation='tanh'),
        #                     name='projection', input='dropout')
        #
        # # Output Layer
        # self.model.add_output(name='output', input='projection')
        # self.model.compile('rmsprop', {'output': 'mse'})

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    # save as .h5 file
    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def qualitative_CNN(self, vocab_size, emb_dim, max_len, nb_filters):
        self.max_len = max_len
        max_features = vocab_size

        filter_lengths = [3, 4, 5]
        print("Build model...")
        # self.qual_model = Model()
        # self.qual_conv_set = {}
        '''Embedding Layer'''
        self.qual_model.add_input(
            name='input', input_shape=(max_len,), dtype=int)

        self.qual_model.add_node(Embedding(max_features, emb_dim, input_length=max_len, weights=self.model.nodes['sentence_embeddings'].get_weights()),
                                 name='sentence_embeddings', input='input')

        '''Convolution Layer & Max Pooling Layer'''
        for i in filter_lengths:
            model_internal = Sequential()
            model_internal.add(
                Reshape(dims=(1, max_len, emb_dim), input_shape=(max_len, emb_dim)))
            self.qual_conv_set[i] = Convolution2D(nb_filters, i, emb_dim, activation="relu", weights=self.model.nodes[
                                                  'unit_' + str(i)].layers[1].get_weights())
            model_internal.add(self.qual_conv_set[i])
            model_internal.add(MaxPooling2D(pool_size=(max_len - i + 1, 1)))
            model_internal.add(Flatten())

            self.qual_model.add_node(
                model_internal, name='unit_' + str(i), input='sentence_embeddings')
            self.qual_model.add_output(
                name='output_' + str(i), input='unit_' + str(i))

        self.qual_model.compile(
            'rmsprop', {'output_3': 'mse', 'output_4': 'mse', 'output_5': 'mse'})

    def train(self, X_train, V, item_weight, seed):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        np.random.seed(seed)
        X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        print("Train...CNN module")
        history = self.model.fit(X_train, V, verbose=0, batch_size=self.batch_size,
                                 epochs=self.nb_epoch, sample_weight=item_weight)

        # cnn_loss_his = history.history['loss']
        # cmp_cnn_loss = sorted(cnn_loss_his)[::-1]
        # if cnn_loss_his != cmp_cnn_loss:
        #     self.nb_epoch = 1
        return history

    def get_projection_layer(self, X_train):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        Y = self.model.predict(X_train, batch_size=len(X_train))
        return Y
