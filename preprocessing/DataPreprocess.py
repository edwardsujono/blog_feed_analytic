import pandas_datareader as web
import pandas as pd
import datetime
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np


class DataPreprocess:

    def __init__(self):

        # for train data

        self.df = pd.read_csv('/Users/edwardsujono/Python_Project/blog_feed_analytic/data/blogData_train.csv')
        # change the index first
        self.df.columns = range(self.df.shape[1])

        # for test data
        self.df_test = pd.read_csv('/Users/edwardsujono/Python_Project/blog_feed_analytic/data/blogData_test-2012.03.31.01_00.csv')
        self.df_test.columns = range(self.df_test.shape[1])

        return

    # based on the analysis
    # [50, 51, 52, 53, 276 ]
    def return_train_data(self, column_list=list([50, 51, 52, 53, 276]), non_zero=True):

        if non_zero:
            df_non_zero = self.df[self.df[280] > 0]
        else:
            df_non_zero = self.df

        df_y = df_non_zero[280].astype(np.int64)
        df_x = df_non_zero[column_list].astype(np.int64)
        df_x = self.normalize_value(df_x)
        # df_x = self.apply_log_func(df_x[df_x >= 0])

        return df_x, df_y

    def return_test_data(self, column_list=list([50, 51, 52, 53, 276]), non_zero=True):

        if non_zero:
            df_non_zero = self.df_test[self.df_test[280] > 0]
        else:
            df_non_zero = self.df_test

        df_test_x = df_non_zero[column_list].astype(np.int64)
        df_test_y = df_non_zero[280].astype(np.int64)
        df_test_x = self.normalize_value(df_test_x)
        # df_test_x = self.apply_log_func(df_test_x)

        return df_test_x, df_test_y

    def normalize_value(self, df):
        df_return = (df - min(df)) / (max(df) - min(df))
        return df_return

    def apply_log_func(self, df):
        df_return = (np.log(df+1))
        return df_return

