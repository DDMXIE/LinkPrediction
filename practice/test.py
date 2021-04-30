import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from decimal import Decimal
from sklearn import metrics
import random

def loadData():
    data_grid = pd.read_csv('./data/USAir.txt', sep='\t', header=None, index_col=False)
    return data_grid

def initAndMergeData(data_grid):
    # use sklearn to divide the dataset
    train, test, train_label, test_lable = train_test_split(data_grid, range(len(data_grid)), test_size=0.1)

    print('训练集 90% 数据：')
    print(train)
    print('测试集 10% 数据：')
    print(test)

    data_grid_set = (train, test)
    return data_grid_set

def getAdjacencyMatrix(data,len):
    # init the adjacent matrix[i][j]
    matrix = [[0] * len for i in range(len)]

    for row in data.values:
        i = row[1] - 1
        j = row[0] - 1
        matrix[i][j] = 1
        matrix[j][i] = 1

    # for row in data.values:
    #     i = row[1]
    #     j = row[0]
    #     matrix[i][j] = 1
    #     matrix[j][i] = 1

    matrix_df = pd.DataFrame(matrix)
    return matrix_df

# 得到原矩阵的反矩阵
def getContraryMatrix(matrix_df):
    matrix = matrix_df
    matrix1 = matrix.replace({1: 2})
    matrix2 = matrix1.replace({0: 1})
    matrix3 = matrix2.replace({2: 0})
    return matrix3

def getNumOfNodes(matrix_df):
    matrix_df_sort = matrix_df.sort_values(by=1)
    max = matrix_df_sort.max()
    return max.max()

def getNumOfEdges(matrix_df):
    return matrix_df.shape[0]

if __name__ == '__main__':
    # 初始化 解析
    data_grid = loadData()
    print(data_grid)

    # sklearn分割数据集
    data_grid_set = initAndMergeData(data_grid)
    # print(data_grid_set)

    # 获得每个类目的训练集和测试集的数据
    data_grid_train = data_grid_set[0]
    data_grid_test = data_grid_set[1]

    # Dataframe化数据集
    data_grid_df = pd.DataFrame(data_grid)
    data_grid_train_df = pd.DataFrame(data_grid_train)

    print('-------------result--------------')

    number_nodes = getNumOfNodes(data_grid)
    print('N: %d' % number_nodes)

    number_edges = getNumOfEdges(data_grid)
    print('M: %d' % number_edges)

    # # 求出邻接矩阵
    # # 总邻接矩阵
    # data_grid_adjMatrix = getAdjacencyMatrix(data_grid_df, 332)
    #
    #
    # # 训练集的邻接矩阵
    # data_grid_train_adjMatrix = getAdjacencyMatrix(data_grid_train,  332)
    # print('-------训练集的邻接矩阵-------')
    # print(data_grid_train_adjMatrix)
    #
    #
    # # 测试集的邻接矩阵
    # data_grid_test_adjMatrix = getAdjacencyMatrix(data_grid_test, 332)
    # print('-------测试集的邻接矩阵-------')
    # print(data_grid_test_adjMatrix)
    #
    # # 求出原邻接矩阵的反矩阵
    # print('-------原邻接矩阵-------')
    # print(data_grid_adjMatrix)
    # data_grid_contrary = getContraryMatrix(data_grid_adjMatrix)
    # print('-------原邻接矩阵的反矩阵-------')
    # print(data_grid_contrary)