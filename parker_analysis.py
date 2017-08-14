import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # used for plot interactive graph. I like it most for plot

class ParkerAnalysis:

    def __init__(self):
        pass

    def run(self):
        pdata = pd.read_csv("./data/parker_sleeping.csv", header=0)
        pdata.drop("Day", axis=1, inplace=True)
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
        self.do_correlation_matrix(pdata, data_cols)

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


if __name__ == "__main__":
    pa = ParkerAnalysis()
    pa.do_correlation_matrix()