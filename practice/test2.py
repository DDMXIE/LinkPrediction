import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from decimal import Decimal
from sklearn import metrics
import random

def computeA(matrix_df):
    matrix_df_1 = matrix_df * 0.001
    print(matrix_df_1)
    print(matrix_df + (matrix_df * 0.001))

if __name__ == '__main__':
    A = [[1, 0, 0], [0, 2, 2], [0, 2, 2]]
    df_A = pd.DataFrame(A)
    computeA(df_A)
