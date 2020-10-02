from math import sqrt

import keras
from keras import Sequential
from keras.layers import Dropout, Dense
from keras import initializers
from keras import optimizers
from keras import regularizers
import pickle


class Network:

    def __init__(self, size_of_state, nr_of_actions, learning_rate=0.05, epochs=10, batch_size=32):
        self.size_of_state = size_of_state
        self.nr_of_actions = nr_of_actions
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(size_of_state,),
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=1/sqrt(size_of_state), seed=None)
                        , kernel_regularizer=regularizers.l2(0.005)))
        model.add(Dropout(rate=0.5))
        model.add(Dense(32, activation='relu',
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=1/8, seed=None)
                        , kernel_regularizer=regularizers.l2(0.005)))
        model.add(Dropout(rate=0.5))
        model.add(Dense(nr_of_actions, activation='linear',
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=1/sqrt(32), seed=None)
                        , kernel_regularizer=regularizers.l2(0.005)))
        print(model.summary())
        model.compile(optimizer=optimizers.SGD(lr=self.learning_rate, nesterov=True),
                      loss='mse', metrics=['accuracy'], )
        self.model = model

    def result(self, input_state):
        # return self.model.predict(input_state).tolist()
        return self.model.predict(input_state)

    def train(self, training_vector, target_vector):
        self.model.fit(training_vector, target_vector, self.batch_size, self.epochs, 0)

    def save_model(self):
        from datetime import datetime
        today = datetime.today()
        fileName = 'model_' + str(today.day) + '_' + str(today.month) + '_' + \
                   str(today.hour) + '_' + str(today.minute)

        with open(fileName, 'wb') as filex:
            pickle.dump(self, filex)

    def load_model(self, fileName):
        with open(fileName, 'rb') as filex:
            object = pickle.load(filex)

        self.model = object.model
        self.learning_rate = object.learning_rate
        self.size_of_state = object.size_of_state
        self.nr_of_actions = object.nr_of_actions
        self.epochs = object.epochs
        self.batch_size = object.batch_size

        print(self.model.summary())


if __name__ == '__main__':
    network = Network(20, 5)
