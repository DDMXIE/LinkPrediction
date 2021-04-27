import pandas as pd
import numpy as np

# This is a sample Python script.
def computeMatrix():
    matrix = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 0, 0]]
    df_matrix = pd.DataFrame(matrix)
    print(df_matrix)
    print('\n')
    print(df_matrix.T)
    # multi_matrix = np.dot(df_matrix.T, df_matrix)
    multi_matrix = np.dot(df_matrix, df_matrix.T)
    print('\n')
    print(multi_matrix)

if __name__ == '__main__':
    computeMatrix()
