import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # used for plot interactive graph. I like it most for plot
from sklearn import preprocessing

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

    def do_correlation_matrix(self):
        pdata = pd.read_csv("./data/parker_sleeping.csv", header=0)
        pdata.drop("Day", axis=1, inplace=True)
        pdata.drop("Counter", axis=1, inplace=True)
        data_cols = list(pdata.columns[0:8])
        corr = pdata[data_cols].corr()  # .corr is used for find corelation
        plt.figure(figsize=(14, 14))
        sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.2f', annot_kws={'size': 16},
                    xticklabels=data_cols, yticklabels=data_cols,
                    cmap='coolwarm')
        plt.show()

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


if __name__ == "__main__":
    pa = ParkerAnalysis()
    pa.do_data_scaling_an_normalization()