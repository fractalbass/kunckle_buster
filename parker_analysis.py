import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # used for plot interactive graph. I like it most for plot
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Activation
from keras.models import model_from_json
from keras import optimizers
from keras import backend as K
from keras.utils import plot_model


class ParkerAnalysis:

    def __init__(self):
        pass

    def run(self):
        pdata = pd.read_csv("./data/parker_sleeping.csv", header=0)
        #pdata = pd.read_csv("./data/data.csv")

        pdata.drop("id", axis=1, inplace=True)
        pdata.drop("Counter", axis=1, inplace=True)
        print(pdata)
        data_cols = list(pdata.columns[1:8])
        print(pdata.head())
        print(pdata.describe())
        fig, ax = plt.subplots()
        ax.axis((0,6,0,3000))
        ax.plot(pdata)
        ax.legend(pdata.columns.values.tolist())
        plt.show()

    def do_correlation_matrix(self, drop_cols):
        pdata = pd.read_csv("./data/data.csv", header=0)
        for d in drop_cols:
            pdata.drop(d, axis=1, inplace=True)

        data_cols = list(pdata.columns[0:11])
        corr = pdata[data_cols].corr()  # .corr is used for find corelation
        g = sns.clustermap(corr, cbar=True, square=True, annot=True, fmt='.2f', annot_kws={'size': 8},
                    cmap='coolwarm', figsize=(8, 8))
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        plt.show()
        g.savefig("pretty_map")

    def do_data_scaling_an_normalization(self):
        pdata = pd.read_csv("./data/parker_sleeping.csv", header=0)
        pdata.drop("Day", axis=1, inplace=True)
        pdata.drop("Counter", axis=1, inplace=True)
        pdata.drop("Bed", axis=1, inplace=True)
        pdata.drop("Sunshine",  axis=1, inplace=True)
        print("Data:")
        print(pdata)

        pretty_printer = lambda x: str.format('{:.2f}', x)

        nd_normalized = preprocessing.normalize(pdata, norm="l2")


        min_max_scaler = preprocessing.MinMaxScaler()
        nd_scaled = min_max_scaler.fit_transform(pdata)

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        ax1.axis((0, 6, 0, 3000))
        ax1.set_title("Raw Data")
        ax1.plot(pdata)

        ax2.set_title("Normalized")
        ax2.axis((0,6,0, 0.2))
        ax2.plot(nd_normalized)

        ax3.set_title("Scaled")
        ax3.axis((0, 6, 0, 1))
        ax3.plot(nd_scaled)

        ax1.legend(pdata.columns.values.tolist())
        ax2.legend(pdata.columns.values.tolist())
        ax3.legend(pdata.columns.values.tolist())

        plt.show()

    def preprocess_data(self, pdata, preserve):
        preserved = pdata["{0}".format(preserve)]
        pdata.drop("{0}".format(preserve), axis=1, inplace=True)

        nd_normalized = preprocessing.normalize(pdata, norm="l2")
        min_max_scaler = preprocessing.MinMaxScaler()
        nd_scaled = min_max_scaler.fit_transform(nd_normalized)

        # fig, (ax1,ax2,ax3) = plt.subplots(3)
        # box1 = ax1.get_position()
        # ax1.set_title("Raw Data")
        # ax1.plot(pdata)
        #
        # box2 = ax2.get_position()
        # ax2.set_title("Normalized")
        # ax2.plot(nd_normalized)
        #
        # box3 = ax3.get_position()
        # ax3.set_title("Scaled")
        # ax3.plot(nd_scaled)
        #
        # plt.show()
        preprocessed_data = pd.DataFrame(data=nd_scaled, columns=pdata.columns, dtype='float')
        preprocessed_data["{0}".format(preserve)] = preserved.values

        return preprocessed_data


    def do_machine_learning_random_forest(self):
        data = pd.read_csv("./data/data.csv", header=0)
        data.drop("Unnamed: 32", axis=1, inplace=True)
        data.drop("id", axis=1, inplace=True)
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        data = self.preprocess_data(data, preserve="diagnosis")
        prediction_var = ['fractal_dimension_mean',
                          'smoothness_mean',
                          'symmetry_mean',
                          'radius_mean',
                          'texture_mean',
                          'compactness_mean']

        train, test = train_test_split(data, test_size=0.3)
        train_X = train[prediction_var]
        train_y = train.diagnosis
        test_X = test[prediction_var]
        test_y = test.diagnosis

        model = RandomForestClassifier(n_estimators=100)

        model.fit(train_X, train_y.astype(int))

        prediction = model.predict(test_X)
        accuracy = metrics.accuracy_score(prediction, test_y)
        plt.show()
        print("Calculation complete.  Random Forest Accuracy: {0}".format(accuracy))
        return accuracy


    def do_neural_network_estimation(self):
        data = pd.read_csv("./data/data.csv", header=0)
        data.drop("Unnamed: 32", axis=1, inplace=True)
        data.drop("id", axis=1, inplace=True)
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': -1})
        data = self.preprocess_data(data, "diagnosis")
        prediction_var = ['fractal_dimension_mean',
                          'smoothness_mean',
                          'symmetry_mean',
                          'radius_mean',
                          'texture_mean',
                          'compactness_mean']
        train, test = train_test_split(data, test_size=0.3)

        train_X = train[prediction_var]
        train_y = train.diagnosis
        test_X = test[prediction_var]
        test_y = test.diagnosis

        model = Sequential()
        model.add(Dense(6, input_shape=(6,), activation='relu'))
        model.add(Dense(100, activation='softmax'))
        model.add(Dense(1, activation='tanh'))
        model.compile(optimizer='rmsprop',
                      loss='mean_squared_error')
        model.fit(train_X.values, train_y.values, batch_size=100, epochs=2000, verbose=False)

        correct_m = 0
        correct_b = 0
        type_I = 0
        type_II = 0

        for x in range(0,len(test_X.values)):

            i = np.reshape(test_X.values[x], (1, 6))
            estimate = model.predict(i)

            if estimate > 0 and test_y.values[x] > 0:
                correct_m = correct_m + 1
            if estimate < 0 and test_y.values[x] < 0:
                correct_b = correct_b + 1
            if estimate > 0 and test_y.values[x] < 0:
                type_I = type_I + 1
            if estimate < 0 and test_y.values[x] > 0:
                type_II = type_II + 1

        a = (correct_m + correct_b) / len(test_X.values)
        print("\n\n{0},{1},{2},{3},{4}\n\n".format(correct_b, correct_m, type_I, type_II, a))

if __name__ == "__main__":
    pa = ParkerAnalysis()
    #pa.do_correlation_matrix(["id", "Unnamed: 32"])

    #for x in range(0,500):

    print(pa.do_machine_learning_random_forest())

    #print("correct_b, correct_m, type_I, type_II, a")
    #for x in range(0,25):
    #    pa.do_neural_network_estimation()