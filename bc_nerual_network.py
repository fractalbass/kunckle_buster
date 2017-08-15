import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Activation
from keras.models import model_from_json
from keras import optimizers
from keras import backend as K
import matplotlib.pyplot as plt

class BreastCancerNetwork:


    def __init__(self, training_file, test_file):
        self.training_file = training_file
        self.test_file = test_file
        model = Sequential()
        model.add(Dense(100, input_shape=(5,)))
        model.add(Dense(100, activation='sigmoid'))
        model.add(Dense(100, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #agd = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        model.compile(optimizer='sgd',
                      loss='mean_squared_error')
        model.summary()
        self.model = model
        self.plot = None
        self.results = None

    def train(self):
        df = pd.read_csv(self.training_file, delim_whitespace=True)
        training_set = df.values[:, 0:5]
        target_set = df.values[:, -1]
        self.results = self.model.fit(training_set, target_set, batch_size=25, epochs=2500)

    def save_model(self, filename):
        # serialize model to JSON
        model_json = self.model.to_json()
        json_name = "{0}.json".format(filename)
        h5_name = "{0}.h5".format(filename)
        with open(json_name, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(h5_name)
        print("Saved model to disk")

    def load_model(self,filename):
        # load json and create model
        json_file = open('{0}.json'.format(filename), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("{0}.h5".format(filename))
        print("Loaded model from disk")

    def display_training_results(self):
        plt.plot(self.results.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show(block=True)

    def predict(self):
        df = pd.read_csv(self.test_file, delim_whitespace=True)
        t = 0.0
        s = 0.0
        for index, row in df.iterrows():
            t = t + 1
            array = row[1:6]
            o = self.model.predict(array.values.reshape(1, 5))
            if o[0][0] < 0.5:
                print("{0},{1}".format(int(row[0]), int(0)))
            else:
                s = s + 1
                print("{0},{1}".format(int(row[0]), int(1)))
        return s/t