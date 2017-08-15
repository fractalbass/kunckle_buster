import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # used for plot interactive graph. I like it most for plot


class ParkerSortedCorrelationMatrix:

    def do_correlation_matrix(self):
        pdata = pd.read_csv("./data/parker_sleeping.csv", header=0)
        pdata.drop("Day", axis=1, inplace=True)
        pdata.drop("Counter", axis=1, inplace=True)
        data_cols = list(pdata.columns[0:8])
        corr = pdata[data_cols].corr()
        plt.figure(figsize=(14, 14))
        sns.clustermap(corr, cbar=True, square=True, annot=True, fmt='.2f', annot_kws={'size': 16},
                    xticklabels=data_cols, yticklabels=data_cols,
                    cmap='coolwarm')
        plt.show()

if __name__ == '__main__':
    pscm = ParkerSortedCorrelationMatrix()
    pscm.do_correlation_matrix()